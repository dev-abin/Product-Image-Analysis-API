# Product Image Analysis API

FastAPI learning project for bounded image processing and async Gemini analysis.

```text
Upload -> local quality checks -> Gemini facts -> code decision -> JSON
```

## Quick start

Python 3.12; run from the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env_example .env
```

Set `GEMINI_API_KEY` in `.env`, then start the development server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --no-access-log
```

Open [Swagger UI](http://127.0.0.1:8000/docs) to upload an image. Images that pass
local checks are sent to Gemini; a valid key, available quota, and network access
are required, and provider usage may incur charges.

## API

- `GET /health`: app health only; does not check Gemini.
- `POST /upload`: one multipart field named `file`.
- Response: `status` (`pass`, `fail`, `needs_review`), `reasons`, `scores`, `facts`.
- Local checks: EXIF orientation, minimum side of 1000 px, sharpness score >= 50.
  Failed checks skip Gemini. Code checks single product, clean packshot, then
  confidence; confidence below 0.6 requires review.
- Business outcomes return HTTP 200. Provider/output errors produce `needs_review`;
  capacity exhaustion or pre-call waiting deadlines return 503 with `Retry-After: 1`.

## Implementation

- [settings.py](settings.py): validated startup settings; environment overrides `.env`.
  See [.env_example](.env_example) for configuration.
- [main.py](main.py): routes and application lifecycle; owns logging startup/shutdown.
- [validation_pool.py](validation_pool.py): bounded CPU workers using the supplied log queue.
- [gemini_resources.py](gemini_resources.py): one resource class and one lifecycle context
  manager for the shared client, concurrency limits, admission, and optional RPM pacing.
- [analyse_image.py](analyse_image.py): strict facts, deadlines, explicit retry classification,
  and at most one transient retry with jitter. [schemas.py](schemas.py) defines the models.
- [request_logging.py](request_logging.py): request IDs, timing, and queued JSON logs to
  stderr and rotating `logs/app.log`; full queues silently drop new records.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use generated images and mocked Gemini calls; they verify behaviour, not
live provider connectivity, model accuracy, or production throughput.

## Limitations

No authentication, inbound rate limiting, upload-size limit, or durable jobs/results.
Uploads enter memory before admission. Capacity and RPM limits apply per API process;
Gemini's total deadline covers analysis, not the entire upload. Multiple API processes
need coordinated provider quotas and separate log destinations. Keep `.env` private.
