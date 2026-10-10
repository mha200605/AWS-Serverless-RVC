"""GPU batch worker: download the selected model + audio, run Applio inference, validate, upload.

Entry point (container): `python -m worker.worker`  with env JOB_ID and BUCKET set by the API Lambda.
Exit code 0 only when the converted audio was validated and uploaded; anything else exits 1 so AWS
Batch marks the job FAILED. Applio's own exit code is NOT trusted: its `convert_audio` swallows
exceptions and the CLI still exits 0 (seen in source), so success is decided by validating the output.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from rvc_common import keys
from rvc_common.errors import RvcError
from rvc_common.model_cache import ModelCache
from rvc_common.model_source import ModelDownloader
from rvc_common.net import FetchPolicy, SafeFetcher, resolve_public_ips
from rvc_common.redaction import configure_logging, redact, register_secret
from rvc_common.settings import Settings
from rvc_common.validation import parse_job_request, validate_job_id

from .audio import AudioTools

log = logging.getLogger("rvc.worker")

ENV_ALLOWLIST = ("PATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "VIRTUAL_ENV", "PYTHONPATH", "TZ")
ENV_PREFIX_ALLOWLIST = ("NVIDIA_", "CUDA_")
MAX_OUTPUT_BYTES = 300 * 1024 * 1024
SILENCE_DB = -60.0
ERROR_MARKER = "An error occurred"        # printed by Applio's convert_audio on failure (seen in source)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_not_found(exc: Exception) -> bool:
    code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
    return code in ("404", "NoSuchKey", "NotFound")


class TokenProvider:
    """Lazily reads the Hugging Face token from Secrets Manager; kept in memory only."""

    def __init__(self, secret_arn: str, client_factory: Callable[[], Any]):
        self.secret_arn, self.factory, self._token = secret_arn, client_factory, None

    def __call__(self) -> Optional[str]:
        if not self.secret_arn:
            return None
        if self._token is None:
            try:
                raw = self.factory().get_secret_value(SecretId=self.secret_arn)["SecretString"].strip()
                if raw.startswith("{"):
                    obj = json.loads(raw)
                    raw = obj.get("token") or obj.get("HF_TOKEN") or ""
            except Exception:
                raise RvcError("SECRET_UNAVAILABLE", "Could not read the configured Hugging Face token.",
                                http_status=500)
            if not raw:
                raise RvcError("SECRET_UNAVAILABLE", "The configured Hugging Face token is empty.", http_status=500)
            register_secret(raw)
            self._token = raw
        return self._token


class StatusWriter:
    def __init__(self, s3, bucket: str, job_id: str):
        self.s3, self.bucket, self.key = s3, bucket, keys.status_key(job_id)
        self.doc = {"jobId": job_id, "state": "RUNNING", "stage": "STARTING", "startedAt": _now()}

    def update(self, *, must_succeed: bool = False, **fields) -> None:
        self.doc.update(fields)
        self.doc["updatedAt"] = _now()
        try:
            self.s3.put_object(Bucket=self.bucket, Key=self.key, Body=json.dumps(self.doc).encode(),
                               ContentType="application/json")
        except Exception:
            log.exception("could not write status.json")
            if must_succeed:
                raise


@dataclass
class Deps:
    s3: Any
    secrets_factory: Callable[[], Any] = None
    transport: Any = None
    resolver: Any = None
    runner: Callable = subprocess.run
    audio: AudioTools = None
    cache: Optional[ModelCache] = None
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    python_exe: str = sys.executable
    environ: dict = field(default_factory=lambda: dict(os.environ))


def build_infer_command(python_exe, input_wav, output_wav, pth, index, options) -> list:
    """Flags use hyphens: that is what the click-based CLI in core.py declares (docs still show underscores)."""
    for p in (input_wav, output_wav, pth, index or ""):
        if "trained" in str(p):          # Applio rewrites 'trained' -> 'added' inside index paths
            raise RvcError("INTERNAL_ERROR", "Unsafe local path.", http_status=500)
    cmd = [python_exe, "core.py", "infer",
           "--input-path", str(input_wav), "--output-path", str(output_wav), "--pth-path", str(pth),
           "--index-path", str(index) if index else "",
           "--pitch", str(options["pitch"]),
           "--index-rate", str(options["index_rate"] if index else 0),
           "--volume-envelope", str(options["volume_envelope"]),
           "--protect", str(options["protect"]),
           "--f0-method", options["f0_method"],
           "--embedder-model", "contentvec",
           "--export-format", "WAV",
           "--sid", str(options["sid"])]
    if options["split_audio"]:
        cmd += ["--split-audio", "True"]
    if options["clean_audio"]:
        cmd += ["--clean-audio", "True", "--clean-strength", str(options["clean_strength"])]
    return cmd


def scrubbed_env(environ: dict, home: Path) -> dict:
    """Applio unpickles user-supplied checkpoints: give it no AWS/HF credentials or tokens."""
    env = {k: v for k, v in environ.items()
           if k in ENV_ALLOWLIST or k.startswith(ENV_PREFIX_ALLOWLIST)}
    env.update({"HOME": str(home), "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    return env


def run_job(settings: Settings, job_id: str, deps: Deps) -> dict:
    validate_job_id(job_id)
    status = StatusWriter(deps.s3, settings.bucket, job_id)
    workdir = Path(settings.workdir_root) / job_id
    timings, t0 = {}, deps.clock()
    audio = deps.audio or AudioTools()
    try:
        workdir.mkdir(parents=True, exist_ok=True, mode=0o700)
        status.update(stage="LOADING_JOB")
        try:
            body = deps.s3.get_object(Bucket=settings.bucket, Key=keys.job_key(job_id))["Body"].read()
            request = parse_job_request(json.loads(body)["request"], settings)   # re-validate: never trust stored data
        except RvcError:
            raise
        except Exception as exc:
            if _is_not_found(exc):
                raise RvcError("JOB_NOT_FOUND", "Job definition is missing.")
            raise RvcError("INVALID_JOB", "Job definition could not be read.")

        # ---- model (only the selected files) --------------------------------------------------
        status.update(stage="DOWNLOADING_MODEL")
        t = deps.clock()
        tokens = TokenProvider(settings.hf_token_secret_arn, deps.secrets_factory or (lambda: None))
        downloader = ModelDownloader(settings, transport=deps.transport, resolver=deps.resolver, cache=deps.cache,
                                     clock=deps.clock, sleep=deps.sleep)
        model = downloader.fetch_model(request.model, workdir / "model", tokens)
        timings["modelDownloadSeconds"] = round(deps.clock() - t, 2)

        # ---- input audio -----------------------------------------------------------------------
        status.update(stage="DOWNLOADING_INPUT")
        raw_input = workdir / "input.raw"
        if request.input_url:
            fetcher = SafeFetcher(FetchPolicy(allowed_hosts=settings.allowed_input_hosts,
                                              redirect_suffixes=settings.allowed_input_hosts, token_hosts=(),
                                              socket_timeout=settings.socket_timeout_seconds,
                                              total_timeout=settings.download_timeout_seconds),
                                  transport=deps.transport, resolver=deps.resolver or resolve_public_ips,
                                  clock=deps.clock, sleep=deps.sleep)
            fetcher.fetch(request.input_url, raw_input, max_bytes=settings.max_input_bytes)
        else:
            try:
                head = deps.s3.head_object(Bucket=settings.bucket, Key=keys.input_key(job_id))
            except Exception as exc:
                if _is_not_found(exc):
                    raise RvcError("INPUT_MISSING", "No uploaded audio was found for this job.")
                raise
            if head["ContentLength"] > settings.max_input_bytes:
                raise RvcError("INPUT_TOO_LARGE", "Uploaded audio exceeds the size limit.")
            deps.s3.download_file(settings.bucket, keys.input_key(job_id), str(raw_input))

        # ---- normalise input -------------------------------------------------------------------
        status.update(stage="PREPARING_AUDIO")
        info = audio.probe(raw_input)
        if info.duration <= 0 or info.duration > settings.max_input_seconds:
            raise RvcError("INPUT_TOO_LONG", f"Audio must be between 0 and {settings.max_input_seconds} seconds.")
        input_wav, output_wav = workdir / "input.wav", workdir / "output.wav"
        audio.to_wav(raw_input, input_wav, settings.max_input_seconds)
        raw_input.unlink(missing_ok=True)

        # ---- inference -------------------------------------------------------------------------
        status.update(stage="CONVERTING")
        t = deps.clock()
        cmd = build_infer_command(deps.python_exe, input_wav, output_wav, model.pth_path, model.index_path,
                                  request.options)
        try:
            proc = deps.runner(cmd, cwd=settings.applio_dir, env=scrubbed_env(deps.environ, workdir),
                               capture_output=True, text=True, timeout=settings.infer_timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            raise RvcError("INFERENCE_TIMEOUT", "Voice conversion exceeded the time limit.")
        timings["inferenceSeconds"] = round(deps.clock() - t, 2)
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log.info("applio exit=%s tail=%s", proc.returncode, redact(text[-2000:]))
        if proc.returncode != 0 or ERROR_MARKER in text:
            raise RvcError("INFERENCE_FAILED", "Voice conversion failed. Check the model files and options.")

        # ---- validate output -------------------------------------------------------------------
        status.update(stage="VALIDATING")
        if not output_wav.exists() or output_wav.stat().st_size == 0:
            raise RvcError("OUTPUT_MISSING", "Conversion produced no output.")
        if output_wav.stat().st_size > MAX_OUTPUT_BYTES:
            raise RvcError("OUTPUT_INVALID", "Converted audio is unexpectedly large.")
        out = audio.probe(output_wav)
        lo, hi = info.duration * 0.85 - 1.0, info.duration * 1.15 + 1.0
        if not (lo <= out.duration <= hi):
            raise RvcError("OUTPUT_INVALID", "Converted audio length does not match the input.")
        peak = audio.max_volume_db(output_wav)
        if peak is None or peak < SILENCE_DB:
            raise RvcError("OUTPUT_SILENT", "Converted audio is silent.")

        # ---- upload ----------------------------------------------------------------------------
        status.update(stage="UPLOADING")
        deps.s3.upload_file(str(output_wav), settings.bucket, keys.output_key(job_id),
                            ExtraArgs={"ContentType": "audio/wav"})
        timings["totalSeconds"] = round(deps.clock() - t0, 2)
        status.update(must_succeed=True, state="SUCCEEDED", stage="DONE", finishedAt=_now(), timings=timings,
                      result={"key": keys.output_key(job_id), "sizeBytes": output_wav.stat().st_size,
                              "durationSeconds": round(out.duration, 2), "sampleRate": out.sample_rate},
                      model={"repo_id": request.model.repo_id, "revision": request.model.revision,
                             "pthSha256": model.pth_sha256, "indexSha256": model.index_sha256,
                             "usedIndex": model.index_path is not None, "cacheHits": model.cache_hits})
        return status.doc
    except RvcError as exc:
        log.warning("job %s failed: %s %s", job_id, exc.code, exc.message)
        timings["totalSeconds"] = round(deps.clock() - t0, 2)
        status.update(state="FAILED", finishedAt=_now(), error=exc.to_dict(), timings=timings)
        return status.doc
    except Exception:
        log.exception("job %s crashed", job_id)
        status.update(state="FAILED", finishedAt=_now(), timings=timings,
                      error={"code": "INTERNAL_ERROR", "message": "Unexpected worker error."})
        return status.doc
    finally:
        if settings.delete_input_after_job:
            try:
                deps.s3.delete_object(Bucket=settings.bucket, Key=keys.input_key(job_id))
            except Exception:
                log.warning("could not delete input object for %s", job_id)
        shutil.rmtree(workdir, ignore_errors=True)
        if deps.cache is not None:
            try:
                deps.cache.prune()
            except Exception:
                log.warning("cache prune failed")


def main() -> int:
    configure_logging()
    import boto3  # imported lazily so unit tests need no AWS SDK

    settings = Settings.from_env()
    job_id = os.environ.get("JOB_ID", "")
    cache = (ModelCache(settings.model_cache_dir, settings.model_cache_max_bytes, settings.model_cache_ttl_seconds)
             if settings.model_cache_dir else None)
    deps = Deps(s3=boto3.client("s3"), secrets_factory=lambda: boto3.client("secretsmanager"), cache=cache)
    try:
        doc = run_job(settings, job_id, deps)
    except RvcError as exc:                # invalid job id etc. -- nothing to write status for
        log.error("%s: %s", exc.code, exc.message)
        return 1
    return 0 if doc.get("state") == "SUCCEEDED" else 1


if __name__ == "__main__":
    sys.exit(main())
