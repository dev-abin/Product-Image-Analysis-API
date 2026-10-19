# Product Image Analysis API

A small FastAPI application that checks image quality and uses Google Gemini to
extract structured product facts. Requests include JSON logs, a request ID, and
latency measurements.

## Request flow

```text
Image -> POST /upload -> image validation -> Gemini facts -> code decision -> JSON
                              |
                              +-> validation failure: skip Gemini
```

Image validation applies EXIF orientation and checks:

- Resolution: the shorter side must be at least 1000 pixels.
- Sharpness: Laplacian variance must be at least 50.

Gemini returns facts about the product category, whether a single product and a
clean packshot are visible, where the product is worn, intended room use, and
estimated confidence. These are model estimates rather than verified labels.

## Setup

Use Python 3.12, the version used for local verification. From the project root,
run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env_example .env
```

Edit `.env` and replace the placeholder with your own Gemini API key:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
```

The application loads `.env` from beside `analyse_image.py`. Existing environment
variables take precedence. Keep your real API key out of source control.

Images that pass local validation are sent to Google's Gemini service. A valid
key, network access, and available API quota are required for this step; API usage
may incur charges. Health checks and automated tests don't call Gemini.

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload --no-access-log
```

- Swagger UI: <http://127.0.0.1:8000/docs>
- Health endpoint: <http://127.0.0.1:8000/health>

`--reload` is for local development. `--no-access-log` avoids duplicate Uvicorn
access logs; application request logs remain enabled. Press Ctrl+C to stop.

## Endpoints

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/health` | Returns HTTP 200 with `{"status":"healthy"}`. Confirms the app responds; does not check Gemini connectivity or credentials. |
| POST | `/upload` | Accepts one image as the multipart form field `file`. |

`POST /upload` returns these fields:

| Field | Meaning |
| --- | --- |
| `status` | `pass`, `fail`, or `needs_review` |
| `reasons` | Validation failures or upstream analysis errors |
| `scores` | `min_side` in pixels and `sharpness` score |
| `facts` | Validated Gemini output, or `null` when unavailable |

Example of a local validation failure:

```json
{
  "status": "fail",
  "reasons": ["low_resolution"],
  "scores": {"min_side": 168, "sharpness": 583.38},
  "facts": null
}
```

A completed upload can return HTTP 200 with `status: fail` or `needs_review`;
inspect the response body. Missing multipart fields return HTTP 422. Missing
credentials, provider failures, empty responses, and invalid structured output
produce `needs_review` with `llm_error` and `facts: null`.

The LLM returns facts only, parsed into a strict Pydantic model:

```json
{
  "single_product": true,
  "clean_packshot": true,
  "product_type": "earrings",
  "worn_on": "ear",
  "sits_in_room": false,
  "confidence": 0.95
}
```

`worn_on` must be `ear`, `neck`, `wrist`, `finger`, `face`, or `none`.
`confidence` must be between 0 and 1. All six fields are required; extra fields
and incorrect types are rejected. A clean packshot has a plain background, no
text or watermark, and a product that is not worn by a person.

Code applies these rules in order; the first matching rule wins:

| Condition | Status | Reasons |
| --- | --- | --- |
| Resolution and/or sharpness fails | `fail` | `low_resolution` and/or `blurry` |
| Image cannot be decoded | `fail` | `invalid_image` |
| LLM fails or output is invalid | `needs_review` | `llm_error` |
| `single_product` is false | `fail` | `not_single_product` |
| `clean_packshot` is false | `fail` | `not_packshot` |
| Confidence is below 0.6 | `needs_review` | `low_confidence` |
| Otherwise | `pass` | Empty list |

Confidence exactly 0.6 passes when preceding checks pass. The response contains
only `status`, `reasons`, `scores`, and `facts`; experience routing is omitted.

## Test an image

Start the server first, then select your own JPG, JPEG, PNG, or WebP image in
Swagger UI or Postman. Use an image at least 1000 pixels on both sides to reach
Gemini analysis. Low-resolution or blurry images return validation results
without calling Gemini. Local `images` and `test` folders are excluded from Git.

### Swagger UI

Open `/docs`, expand **POST /upload**, select **Try it out**, choose a file, and
select **Execute**. Read the result under **Server response**. If an embedded
browser's picker doesn't open, use a standalone browser.

### Postman

Send `POST http://127.0.0.1:8000/upload`. Under **Body -> form-data**, add a key
named `file`, change its type from **Text** to **File**, and select your image.
Let Postman generate the multipart `Content-Type` header automatically.

## Logging

Application logs are newline-delimited JSON written to the terminal (`stderr`)
and persisted in `logs/app.log` beside the application. The directory is created
automatically. Each file rotates at 5 MiB, retaining three backups (`app.log.1`
through `app.log.3`). These files are excluded from Git. Each request log contains:

```json
{
  "timestamp": "2026-01-01T00:00:00.000+00:00",
  "level": "INFO",
  "logger": "image_api",
  "event": "request.completed",
  "request_id": "example-request-001",
  "method": "GET",
  "path": "/health",
  "status_code": 200,
  "latency_ms": 1.233
}
```

- Supply `X-Request-ID` to correlate client and server logs, or let the app
  generate a UUID. Supplied IDs allow 1-128 ASCII letters, digits, dots,
  underscores, or hyphens. Other values are replaced with a UUID.
- The response includes `X-Request-ID`, including unhandled HTTP 500 responses.
- Server latency measures processing through response completion. Client latency
  includes additional connection and transfer overhead.
- Application code can use `from request_logging import logger` to include the
  active request ID in its logs. `RequestContextFilter` attaches that ID to logs
  from all stages. Unexpected request exceptions include a traceback.
- Stage events cover uploads, validation scores and reasons, skipped Gemini
  calls, provider attempts and retries, LLM duration, and the final business
  verdict. `decision.completed` records `pass`, `fail`, or `needs_review`
  separately from the HTTP status code in `request.completed`.
- LLM errors record the failing stage and exception class, without the provider
  exception message, raw model output, or image contents.
- Request bodies, query strings, and headers are not included in request logs.
  Review logs and tracebacks before sharing them; errors can include local paths.
- Uvicorn startup and diagnostic messages use its own console format.

To follow persisted logs in PowerShell:

```powershell
Get-Content .\logs\app.log -Tail 20 -Wait
```

File rotation is intended for a single application process. For multiple worker
processes or replicas, use console logs with a shared log collector instead of
having workers rotate the same file.

## Run unit tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

To measure line and branch coverage for the application modules:

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=main --cov=analyse_image --cov=validate_image --cov=schemas --cov=request_logging --cov-branch --cov-report=term-missing
```

Tests use generated images and mocked Gemini responses; no API key or external
requests are required. Coverage includes image-quality thresholds, decoding and
EXIF orientation, strict fact validation, decision-rule precedence, provider
timeouts and retry limits, upload responses and file cleanup, request IDs,
JSON logs, exception handling, and concurrent request-context isolation.
These checks verify application behavior, not the real model's image accuracy
or live provider connectivity.

## Repository contents

| File | Purpose |
| --- | --- |
| `main.py` | FastAPI routes and middleware wiring |
| `validate_image.py` | Local resolution and blur checks |
| `analyse_image.py` | Gemini prompt, structured extraction, and API error handling |
| `schemas.py` | Shared Pydantic facts, scores, and response models |
| `request_logging.py` | JSON logging and request tracing middleware |
| `tests/` | Pytest unit tests and upload-route checks |
| `pytest.ini` | Test discovery and import configuration |
| `requirements-dev.txt` | Application dependencies plus pytest and coverage tools |
| `requirements.txt` | Direct dependencies pinned to the locally verified versions |
| `.env_example` | Placeholder environment configuration |
| `.gitignore` | Excludes local secrets, environments, images, logs, and generated files |
| `README.md` | Setup, usage, and limitations |

Local images, screenshots, `.env`, virtual environments, and caches stay on your
machine. No sample images or real credentials are required in the repository.

## Current limitations

This is a local prototype. It has no authentication, rate limiting, or upload-size
limit. Uploads are read into memory, and synchronous image processing and Gemini
calls run within the async route. Add suitable controls and concurrency handling
before exposing it as a public service. Image bytes and results are not saved by
the application.

The model is currently configured as `gemini-flash-latest` in `analyse_image.py`.
Each Gemini attempt has a 30-second timeout, with one application-level retry
for supported transient failures. Empty or invalid model output returns
`needs_review` without a retry.
