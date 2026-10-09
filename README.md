# AI in Practice Lab

This repository contains the Labs 1–7 exercises and the Aurora policy assistant
capstone. Lab 7 serves grounded policy answers through FastAPI, shows expandable
citations in Streamlit, records traces and cost, and runs an offline golden-set
regression gate.

## Quick start

Use Python 3.11–3.14. From the repository root in PowerShell:

> Lab 7 uses its own compact, committed offline cache at
> `labs/lab7/offline_cache/calls.sqlite3`; it is built from genuine responses
> consumed by the golden gate, with no API key required during CI. The larger
> root `.aip_cache` remains the interactive development cache.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:AIP_OFFLINE = "1"
python -m uvicorn labs.lab7.service:app --port 8000
```

The offline mode needs the committed `.aip_cache/calls.sqlite3` model-call
cache. It returns a clear cache-miss error rather than contacting a provider.
With the service running, open a second terminal in the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run labs/lab7/ui.py
```

The service API is documented in [the Lab 7 brief](labs/lab7/README.md).
Operational traces are shown by `python -m streamlit run labs/lab7/dashboard.py`.
To verify the implementation locally:

```powershell
python -m pytest tests/test_lab7_service.py
python -m ruff check labs/lab7/service.py labs/lab7/ui.py labs/lab7/dashboard.py labs/lab7/gate.py
$env:AIP_OFFLINE = "1"
$env:AIP_CACHE_DIR = "labs\lab7\offline_cache"
python labs/lab7/gate.py
```

On an already provisioned machine, launching the service and UI takes only a
few commands. A clean installation may take longer than five minutes because
the retrieval and model dependencies are sizeable.

When refreshing offline gate assets after an intentional evaluation change,
populate the development cache while online, then run
`python labs/lab7/prepare_offline_cache.py`; it refuses to contact providers
and copies only genuine cache entries actually read by a full gate measurement.
