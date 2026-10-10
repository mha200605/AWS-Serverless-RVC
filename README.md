# rvc-serverless

Asynchronous voice conversion with **Applio (RVC)** on **scale-to-zero GPU AWS Batch**.
Models are pulled from **your Hugging Face repository**, chosen **per request**; nothing model-related is
uploaded by you or baked into the image.

> **Status: local-only. Nothing is deployed, built into an image, or run against real Applio.**
> The 139 unit tests use fakes (no network, no AWS, no real models). `template.yaml` is parsed and statically
> checked by tests but has **not** been validated by `sam validate` / `cfn-lint` or by AWS.

## How it works

```
client --SigV4--> API Gateway (HTTP API, IAM auth) --> Lambda (api/handler.py)
   1. POST /jobs            -> {jobId, upload{url,fields}}          (nothing billable yet)
   2. POST <upload.url>     -> audio goes straight to private S3     (size-capped presigned POST)
   3. POST /jobs/{id}/start -> Lambda submits the AWS Batch job
                               Batch starts ONE g4dn.xlarge (min vCPU 0, max 4) -> worker container:
                               download selected model -> download audio -> ffmpeg normalise
                               -> `python core.py infer ...` -> validate output -> upload -> status.json
   4. GET  /jobs/{id}       -> QUEUED | RUNNING(stage) | SUCCEEDED(+5-min signed URL) | FAILED(error code)
```

Expect **minutes** of cold start (instance launch + image pull); this is a job API, not real time.

### Request

```json
POST /jobs
{
  "model": {
    "repo_id": "your-user/your-rvc-models",
    "pth": "voices/alice/alice.pth",
    "index": "voices/alice/alice.index",
    "revision": "<40-char commit sha>"
  },
  "options": {"pitch": 0, "index_rate": 0.3, "protect": 0.33, "f0_method": "rmvpe"}
}
```

* `index` is optional. `repo_id` may be omitted if `DefaultHfRepo` is set.
* Alternatively give direct links: `"model": {"url": "https://huggingface.co/<owner>/<repo>/resolve/<rev>/<file>.pth",
  "index_url": "..."}`. A URL is **never fetched as given**: it is parsed to (repo, revision, file), checked
  against the allow-list, and the download URL is rebuilt from those parts.
* Optional `pth_sha256` / `index_sha256` pin file contents. `input_url` (HTTPS, allow-listed hosts only) is
  disabled unless `AllowedInputHosts` is set; the default flow is the presigned upload.
* Options are range-checked; unknown fields are rejected. The embedder is fixed to `contentvec` so a request
  cannot make Applio download arbitrary models.

```bash
# SigV4 with curl >= 7.75 (use a scoped IAM identity, never root)
curl --aws-sigv4 "aws:amz:REGION:execute-api" --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
     -H "x-amz-security-token: $AWS_SESSION_TOKEN" -H 'content-type: application/json' \
     -d @request.json "$API_URL/jobs"
# upload: POST every field returned in upload.fields, then the file as the LAST multipart field named "file"
```

## Hugging Face access and download hardening

| Concern | What the code does (all covered by tests) |
|---|---|
| Which repos | `AllowedHfRepos` is **required with no default** (`owner/repo` or `owner/*`); empty = nothing allowed. Re-checked in the worker. |
| Which hosts | First hop must be in `ALLOWED_DOWNLOAD_HOSTS` (default `huggingface.co`); redirects only to `*.hf.co` / `*.huggingface.co`. https and port 443 only, no URL credentials, no IP literals. |
| SSRF | Every hop is DNS-resolved; **all** addresses must be globally routable (blocks 10/8, 127/8, 169.254.169.254, link-local, ULA ...). The TCP connection is pinned to the validated IP, so DNS rebinding cannot swap it. ffmpeg runs with `-protocol_whitelist file` and only common audio containers are accepted. |
| Token | Secrets Manager (`HfTokenSecretArn`), read at run time by the worker's job role, held in memory only. Sent only to the first-hop HF host, never to CDN redirects. Never in env vars, the image, logs, `status.json`, API responses or the Applio subprocess (its environment is an allow-list with no `AWS_*`/`HF_*`). Logs pass through a redacting formatter. |
| Size / time | `.pth` ≤ `MaxPthMB`, `.index` ≤ `MaxIndexMB` (Content-Length checked up front **and** bytes counted while streaming); per-read socket timeout 30 s, overall deadline 600 s; bounded retries on 5xx/network errors; free-disk check; written to `.part` (0600) then renamed. |
| Content sanity | `.pth` must look like a torch zip/pickle, `.index` must not be HTML or a Git-LFS pointer; local file names are fixed (`model.pth`, `model.index`), so repo file names never reach the filesystem or command line. |

**Security caveat that no filter removes:** `.pth` checkpoints are pickle-based, so loading one executes code
from it. Treat the allow-listed repository as trusted, use a **read-only fine-grained token scoped to that repo**,
pin `revision` to a commit SHA (`RequirePinnedRevision=true`), and keep the job role minimal (it is: three S3
prefixes + one secret).

## Model download per cold start vs. caching on a worker

| | Download per job (default) | Host cache (`EnableHostModelCache=true`) |
|---|---|---|
| Latency | + model download each job (tens of seconds for typical sizes **[assumption, measure it]**) | saves that time on a repeat hit |
| When it helps | always correct | only when several jobs hit the **same instance** inside its idle window |
| Scale-to-zero reality | cold start is dominated by instance launch + image pull, so the extra download is a small share | with sporadic traffic the instance is gone before the next job, so the cache is usually empty anyway |
| Privacy | private model exists only in the job's temp dir, deleted at job end | private model stays on the instance's encrypted EBS until prune/termination |
| Correctness | always the revision you asked for | only commit-SHA-pinned revisions are cached; each reuse re-hashes the file |
| Cleanup | `rmtree` of the work dir in `finally` | entries unused > 24 h deleted, then LRU down to 4 GiB; dir is 0700; volume is deleted when the instance terminates |

Recommendation: leave it **off** for the PoC; turn it on only if you batch many jobs with one pinned model.
Cost of downloading from Hugging Face is network time, not an AWS line item I could verify.

## Privacy and retention (defaults, all parameters)

| Data | Where | Retention |
|---|---|---|
| Uploaded audio | `s3://…/inputs/` | worker deletes it when the job ends; lifecycle expiry `InputRetentionDays`=1 as a backstop |
| Converted audio | `…/outputs/` | `OutputRetentionDays`=3 |
| job.json / status.json | `…/jobs/` | `MetadataRetentionDays`=7 |
| Model files | worker temp dir (or host cache if enabled) | deleted at job end (cache: see above); **not** stored in S3 or the image |
| Logs | CloudWatch | `LogRetentionDays`=7; no audio, tokens or signed URLs are logged |
| Upload / download links | presigned | upload 15 min (size-capped), result 5 min |

Bucket: all public access blocked, owner-enforced, SSE-S3, TLS-only policy. S3 lifecycle runs asynchronously, so
objects can outlive the configured day by a short while **[from AWS behaviour as I know it, not verified here]**.
Use SSE-KMS only if you need key-level audit; it adds cost (decision below).

## Cost guards built into the template

Hard cap of one 4-vCPU On-Demand GPU instance (`MaxVCpus` allows only `4`, `MinvCpus`/`DesiredvCpus` 0, no Spot);
no job retries; per-job timeout; no NAT gateway, Elastic IPs, VPC endpoints, KMS keys, DynamoDB or load balancer
(tests fail if any appears); ECR keeps 3 images; log retention everywhere; throttled API; optional monthly
budget email. The GPU quota (`Running On-Demand G and VT instances`) is currently 0 in this account, so no GPU
instance can start until it is raised; request **4 vCPUs**, no more.

## Develop and test locally (no AWS, no network, no real models)

```bash
pip install -r requirements-dev.txt          # PyYAML only; tests otherwise need just the standard library
python -m unittest discover -s tests -t .    # 139 tests; two use the local ffmpeg to synthesise a 1 s tone
```

Layout: `src/rvc_common/` (validation, safe downloader, cache, redaction), `src/worker/worker.py` (+`audio.py`),
`src/api/handler.py`, `template.yaml`, `Dockerfile.worker`, `tests/`.

## Before you deploy (nothing below has been done)

**Decisions for you**
1. Which Hugging Face repo(s) go in `AllowedHfRepos`, and may I assume they are yours alone (pickle risk)?
2. Token: a fine-grained read-only token for that repo in Secrets Manager (`HfTokenSecretArn`), or public-only?
3. Require commit-SHA revisions (`RequirePinnedRevision=true`) now, or only after testing?
4. API auth: IAM/SigV4 is the default; is a shared-secret authorizer or Cognito preferable for your client?
5. SSE-S3 (default, free) or SSE-KMS (key audit, extra cost)? Retention values acceptable?
6. Model cache on or off; GPU type (`g4dn.xlarge` default vs `g6`/`g5`, more expensive)?
7. Region (the account session defaults to us-east-1) and whether credits cover EC2 GPU, ECR, Secrets Manager
   (~per secret per month, price not verified), CloudWatch and API Gateway.
8. A budget email address and monthly limit.

**Checks required first**
1. Stop using the root identity: create a scoped IAM admin/deploy role + MFA; none of this should be deployed as root.
2. Request the G/VT vCPU quota increase to 4 (currently 0) and wait for approval.
3. Clone Applio at tag 3.6.5 (release page shows commit `55fe0b9`; resolve the full SHA with `git rev-parse`) and
   re-verify against that tag, not `main`: `requirements.txt` pins, the `infer` command's flags
   (`python core.py infer --help`; source uses hyphenated flags, the docs site still shows underscores), the
   `prerequisites` command flags, and the exact `--split-audio` / `--clean-audio` value syntax.
4. Run **one real local conversion** (your own non-sensitive clip, your own model) and confirm: success output,
   what the **failure** output looks like (the worker relies on the "An error occurred" marker and output
   validation), and **whether a missing/empty `.index` works** with `--index-rate 0`. The no-index path is an
   assumption until this is done.
5. Confirm the real redirect hosts Hugging Face uses for your files (e.g. its CDN hostnames) are covered by
   `ALLOWED_REDIRECT_HOST_SUFFIXES`; the defaults (`hf.co`, `huggingface.co`) are my expectation, not verified.
6. Build the image with pinned `TORCH_VERSION`/`TORCHAUDIO_VERSION`/`BOTO3_VERSION`; check its size,
   `nvidia-smi` + `torch.cuda.is_available()` on a GPU host, and that it runs with `docker run --network none`
   (proves no runtime downloads). Verify the ECS GPU-optimised AMI's NVIDIA driver supports CUDA 12.8.
7. `sam validate --lint` / `cfn-lint` the template; confirm the HTTP API IAM-authorizer syntax, the
   Batch launch-template root device name (`/dev/xvda`), and that API Gateway can write the access-log group.
8. Review the Batch job's memory (14000 MiB on a 16 GiB instance) and the 100 GiB root volume against the real image.
9. After any test: confirm zero running instances/volumes/ENIs; empty the bucket before stack deletion.
