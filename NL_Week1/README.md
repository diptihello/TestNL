# AI-Powered Restaurant Recommendation System

A local Streamlit app that recommends Bangalore restaurants from the
[Zomato dataset on Hugging Face](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation).
The dataset is filtered and ranked first. A Groq-hosted LLM then picks the final
results and explains each choice. Without a Groq key the app still works and
shows a deterministic dataset-ranked list.

Design documents: [problem statement](docs/problemStatement.md),
[architecture](docs/architecture.md), [implementation plan](docs/implementation-plan.md),
[edge cases](docs/edge-case.md).

---

## Phase status

| Phase | Name | Status |
|---|---|---|
| 0 | Scaffold and configuration | Done |
| 1 | Dataset ingestion and preprocessing | Done |
| 2 | Domain: models, filtering, ranking | Done |
| 3 | UI without the LLM | Done |
| 4 | LLM module: client, prompts, schema, guard | Done |
| 5 | Integration and result experience | Done |
| 6 | Evaluation, edge cases, hand-off | Not started |

---

## Recommendation experience

Choose a Bangalore location, budget and minimum rating. Optional cuisine and
free-text preferences refine filtering and ranking; if nothing matches, the app
offers clickable suggestions that prefill the form. With `DEMO_MODE=true`, an
offline fake model exercises the AI-result UI. With a Groq key, the guarded
provider ranks candidates; without a key or when a provider error occurs, the
app shows deterministic dataset-ranked results and a clear notice. Dataset facts
always come from repository rows; model text is displayed separately, grounded,
and cached results are marked with their age. Valid extra model recommendations
are available in **Show more recommendations**, without another request.

The sidebar displays rolling request/token usage. Retry is offered only after a
retryable provider failure, bypasses the cache for one request, and respects the
configured minimum submit interval. See [Phase 5 evaluation notes](docs/evaluation.md)
for the UI-state walkthrough.

---

## Phase 4 LLM module

The Groq client is isolated behind an interface, with a deterministic fake for
demo and no-key modes. Prompts are compact and versioned, model output is
validated and grounded against candidate data, and request/token budgets plus
the parsed-result cache are persisted under `data/`. The sidebar's **Test
connection** button makes one small request only when clicked; it reports a
clear configuration error when no key is set.

One live smoke run on `openai/gpt-oss-120b` (three candidates) succeeded:

| Check | Result |
|---|---|
| Connection test | Successful |
| JSON response mode / complete JSON | Supported; finish reason `stop` |
| Prompt tokens | 511 |
| Completion tokens | 509 |
| Total tokens | 1,020 (below the 2,060 request ceiling) |
| Local prompt estimate | 409 tokens (system + user prompt; the user prompt alone was 266) |

Provider usage fields were present. The live smoke is opt-in and runs with
`pytest tests\test_live.py -q -s --run-live`; ordinary tests do not contact Groq.
The smoke used prompt version 1; version 2 subsequently JSON-escaped the
dynamic location and cuisine fields to prevent section injection. That change
is covered by offline prompt tests; no second live request was made.

---

## Requirements

- Windows with PowerShell (the commands below use it; other shells work with path changes).
- Python 3.11 or newer (developed on 3.12). `pip install -e .` fails early on older versions.
- Git.
- About **1.5 GB of free disk** for the one-time ingestion (the raw CSV is about 574 MB
  and is cached by `huggingface_hub`). The processed parquet is small.
- A Groq API key is optional and only needed for AI explanations (Phase 4 onward).

## Setup

Run these from the project folder (`NL_Week1`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env      # then edit .env; GROQ_API_KEY may stay blank
pytest -q
python scripts\ingest.py         # one-time dataset build (add --sample 2000 for a quick run)
streamlit run app.py
```

### If `Activate.ps1` is blocked

PowerShell may refuse to run the activation script. Either allow scripts for the
current window only:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
.\.venv\Scripts\Activate.ps1
```

or skip activation and call the venv's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Configuration

All settings live in `.env` (copied from [.env.example](.env.example), which
lists every variable with its default and a comment). An environment variable
with the same name overrides `.env`. Invalid settings stop the app with a
message that names the variable but never shows its value.

Set `DEMO_MODE=true` to run with a built-in fake model and never call the API.

To pre-warm the tokenizer encoding used for prompt estimates (optional; estimates
fall back safely if it is unavailable), run:

```powershell
python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
```

---

## Security

- **The app has no authentication.** The Streamlit server is bound to `localhost`
  in [.streamlit/config.toml](.streamlit/config.toml). Do not change the address
  or expose the port on a network.
- Error details are never sent to the browser (`client.showErrorDetails = "none"`);
  tracebacks go to the log only.
- The API key is held as a secret string, sanitized on load (whitespace, quotes and
  a leading `Bearer ` are removed), and redacted from every log line and traceback.

Do these once after cloning:

1. Confirm `.env` is ignored by git. The command must print `.env`:

   ```powershell
   git check-ignore .env
   ```

2. Install the pre-commit hook that blocks any commit containing a string that
   starts with `gsk_` (a Groq key):

   ```powershell
   python scripts\install_hooks.py
   ```

   You can run the same check by hand at any time: `python scripts\check_no_secrets.py`.

If a key is ever committed, rotate it in the Groq console straight away; removing
it from history is not enough.

---

## Project layout

```text
app.py                     Streamlit entry point
src/zomato_reco/
  config.py                Settings from .env with validation
  logging_setup.py         single-line JSON logs with key redaction
  data/                    preprocessing (Phase 1) and repository (Phase 2)
  domain/                  models, filters, ranking (Phase 2)
  llm/                     Groq client, prompts, schema, budget guard (Phase 4)
  service/                 recommendation orchestration (Phase 3, 5)
  ui/                      Streamlit components and session state (Phase 3)
scripts/
  ingest.py                one-time dataset download and preprocessing
  check_no_secrets.py      scan tracked files for Groq-style keys
  install_hooks.py         install the pre-commit secret check
tests/                     pytest suite; runs fully offline
data/processed/            restaurants.parquet and manifest.json (after ingestion)
docs/                      problem statement, architecture, plan, edge cases
```

## Running the tests

```powershell
pytest -q                  # offline suite
pytest -q --run-live       # also runs tests marked live (needs GROQ_API_KEY, Phase 4)
```

---

## Dataset statistics

From the first full ingestion ([manifest.json](data/processed/manifest.json)),
built 2026-10-15 from `zomato.csv` at dataset commit
`5738e9eda2fad49ad51c6e0ed26e761d9b947133`.

| Measure | Value |
|---|---|
| Raw rows | 51,717 |
| Restaurants after cleaning | 12,485 |
| Duplicates removed | 39,211 (the source lists one restaurant once per `listed_in` type and city) |
| Rows dropped for empty location | 21 |
| Rows dropped for empty name | 0 |
| Rating parse rate | 100 % of 41,665 numeric-looking values |
| Rated restaurants (after dedupe) | 9,478 (the rest are `NEW` or `-`) |
| Cost parse rate | 100 % of 51,371 non-empty values |
| Distinct locations | 93 |

Cost for two (INR) percentiles, per restaurant:

| p10 | p33 | p50 | p66 | p90 |
|---|---|---|---|---|
| 200 | 300 | 400 | 500 | 850 |

### Budget thresholds (tuned)

The planned defaults of 400 / 900 were far from the p33 / p66 of 300 / 500: they
put 59 % of restaurants in `low` and only 8.5 % in `high`. The defaults are now
`BUDGET_LOW_MAX=300` and `BUDGET_MEDIUM_MAX=500`, giving roughly
41 % / 31 % / 28 %. Updated in `config.py`, `.env.example` and architecture 4.4.

### Parquet commit decision

`data/processed/restaurants.parquet` (0.77 MB) and `manifest.json` **are
committed**, because the parquet is well under the 5 MB limit. Reviewers can run
the app straight after `pip install` without the 574 MB download. Re-run
`python scripts\ingest.py` only to refresh the data; ids are stable across runs.

### Ingestion notes (Windows)

- `huggingface_hub` may warn that symlinks are unsupported. Caching still works
  but uses more disk. Enabling Windows Developer Mode removes the warning.
- It may also suggest installing `hf_xet` for faster downloads. This is optional.
