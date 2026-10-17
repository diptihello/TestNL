# Phase 5 integration evaluation

## Offline UI-state walkthrough

All states below were exercised with the test fixture dataset and no external
provider requests. `DEMO_MODE=true` selects the deterministic fake client for
the AI-result path. The provider-failure path uses a scripted timeout; the
budget path injects a blocked reservation.

| Architecture 8.2 state | Trigger | Evidence / expected rendering |
|---|---|---|
| Not ready | Point `PROCESSED_PATH` at a missing parquet file. | `test_missing_processed_data_shows_ingest_command`: error explains the dataset is not built and shows `python scripts\\ingest.py`. |
| Ready | Load the app with the fixture parquet. | `test_loads_and_location_options`: form is available and has no preselected location; sidebar displays model, usage, and dataset details. |
| Validation error | Submit four comma-separated cuisine values. | `test_invalid_cuisine_is_reported_without_service_call`: validation message appears and the recommender is not called. |
| Submitting | Set the session busy flag while the form is rendered. | `test_busy_ignores_second_submit`: submit button is disabled; search exceptions reset busy in `finally`. |
| Results (AI) | Submit a valid search in demo mode. | `test_submit_blank_cuisine_yields_cards`: AI badge and explanation panel are shown; extra results are expandable when returned. |
| Results (fallback) | Disable demo mode and leave the key blank. | `test_missing_key_shows_fallback_without_ai_column_or_retry`: sorted-by-rating badge and missing-key notice appear without an AI column or retry. |
| No matches | Submit an impossible cuisine/filter combination. | `test_impossible_query_shows_relaxation`: empty state and clickable, current relaxation suggestions are rendered. |
| Budget blocked | Make the guard reject the reservation. | `test_budget_block_is_rendered_without_provider_retry`: wait notice and fallback cards appear; there is no retry button because no provider call failed. |

The retry flow is also covered by `test_retry_button_bypasses_cache_once`:
only a retryable provider failure exposes the button, and the subsequent single
call bypasses the cache. Service tests exercise cache hits while the budget
would block another request, cache id revalidation, timeout accounting,
malformed output, partial outputs, grounding, and the no-key fallback.

## Provider evaluation

The Phase 4 live smoke is documented in the README. On 2026-10-17, five varied
searches were run against the configured provider with cache bypassed:

| Search | Match count | Outcome |
|---|---:|---|
| No cuisine or notes | 384 | AI result |
| Cuisine selected | 46 | Safe fallback |
| Notes, online-order preference | 384 | AI result |
| Different location and notes | 160 | Safe fallback |
| Impossible cuisine | 0 | Empty result; provider not called |

Usage before the run was 6 requests / 1,206 tokens for the day. Afterward it was
10 requests / 7,711 tokens: the four non-empty searches each added one request,
and the impossible search added none. The total remained below the 25,000-token
evaluation budget. Two provider calls produced AI results; the other two were
returned as safe fallbacks. The live run was direct-service validation rather
than a human-scored UI rubric; scoring remains part of Phase 6.

The opt-in one-request smoke test can also be run with a configured key:

```powershell
pytest tests\test_live.py -q -s --run-live
```

That opt-in test makes a single small provider request. The five varied live
searches are recorded above; the human-scored rubric remains part of Phase 6.

Test isolation note: an initial full-suite run exposed that an empty
`GROQ_API_KEY` environment variable is ignored by settings loading and allowed
the configured key to be inherited by the missing-key UI test. One provider
request occurred in that failed run. The test now overrides the variable with
whitespace (which sanitizes to no key), and the final full suite ran offline.
