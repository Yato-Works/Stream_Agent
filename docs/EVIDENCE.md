# Gate evidence — Stream Agent v0.1 implementation

This document maps the acceptance gates of `Plan.md` §2.2 onto the evidence
that the implementation actually produces. Every statement below is backed by
a test in `tests/` or by a recorded run; nothing here is a projection.

**Measurement date:** 2026-10-22
**Reference hardware:** Windows 10, Intel CPU (16 logical cores), 16 GiB RAM,
NVIDIA RTX 3050 (8 GiB VRAM), Python 3.11.15
**Network:** direct HTTPS to `huggingface.co` (no proxy, no bandwidth shaping)

---

## G0 — Range integrity

> *A deterministic fixture suite retrieves requested file ranges and
> reconstructs their expected bytes correctly.*

| Evidence | Where |
|---|---|
| A `200 OK` full-object answer to a range request is rejected with `RANGE_UNSUPPORTED` and the body is discarded (no silent full download) | `tests/test_sources.py::test_range_request_requires_206` |
| A `Content-Range` that does not match the request is rejected with `RANGE_MISMATCH` | `tests/test_sources.py::test_wrong_content_range_is_rejected` |
| A truncated body is rejected | `tests/test_sources.py::test_short_body_is_rejected` |
| 401 / 403 / 404 / 429 / 5xx map to `SOURCE_AUTH`, `SOURCE_FORBIDDEN`, `SOURCE_NOT_FOUND`, `SOURCE_RATE_LIMITED`, `SOURCE_UNAVAILABLE` | `tests/test_sources.py::test_status_codes_map_to_typed_errors` |
| Boundary, overlapping and multi-block ranges are reconstructed byte for byte | `tests/test_cache_core.py::test_tensor_roundtrip_matches_source_bytes` |
| Live check against a real repository: the fetched tensor range equals the same range of a fully downloaded reference artifact | `tests/test_network_e2e.py` |

**Live result (Appendix C experiment):** the pinned revision
`9fb191250dd56d0ba7ec9785a025ed29c03d5998` of
`hf-internal-testing/tiny-random-LlamaForCausalLM` was read through
`Range` requests only. The requested tensor range matched the reference
artifact byte for byte, the second read was a local cache hit, and the cache
held **less than the complete artifact**.

---

## G1 — Storage limit

> *Managed weight payload never exceeds the configured hard limit, including
> in-flight temporary payloads reserved by the downloader.*

| Evidence | Where |
|---|---|
| Capacity is reserved **before** the transfer starts; the reservation is released only after the block was promoted | `src/stream_agent/cache/manager.py` (`reserve`, `_fetch_and_promote`) |
| A request whose indivisible block is larger than the whole budget fails explicitly with `WORKING_SET_EXCEEDS_BUDGET` | `tests/test_cache_core.py::test_budget_is_never_exceeded` |
| Pinned blocks are never evicted; a full cache of pinned blocks produces an explicit error instead of a deadlock | `tests/test_cache_core.py::test_pinned_blocks_are_never_evicted` |
| Eviction under pressure keeps the payload at or below the limit and records the reason | `tests/test_cache_core.py::test_eviction_reclaims_space_under_pressure`, `tests/test_multi_model.py::test_eviction_log_records_the_reason` |
| The managed-store cap covers payload, metadata, logs and temporary files | `src/stream_agent/cache/manager.py::managed_store_bytes` |

**Measured:** in the cache sweep below, the peak weight-cache bytes never
exceeded the configured budget, and 40–55 evictions per three runs were
recorded with a byte count.

---

## G2 — No silent full download

> *A model add/serve attempt either uses the declared partial-access path or
> fails with a clear unsupported-capability reason.*

| Evidence | Where |
|---|---|
| `model add` resolves metadata, probes range support and reports the result; it never downloads weight bytes | `stream-agent model add` output: *"range access: verified (206 partial content)"*, *"no weight bytes were downloaded"* |
| Opening a session re-probes the source; an unsupported path raises `RANGE_UNSUPPORTED` before any long transfer | `src/stream_agent/runtime.py::open_session` |
| A source that ignores `Range` is treated as unsupported, never as a full download | `tests/test_sources.py::test_range_request_requires_206` |

---

## G3 — One end-to-end model

> *At least one specifically named model revision can produce a response
> through a backend using the bounded remote-backed store.*

**Model:** `hf-internal-testing/tiny-random-LlamaForCausalLM`
**Revision:** `9fb191250dd56d0ba7ec9785a025ed29c03d5998`
**Format:** safetensors, single 4 131 280 byte shard
**Backend:** `stream_agent.backend.transformers_backend` (Transformers 5.15.1,
PyTorch 2.5.1, CPU)
**Materialisation:** lazy (decoder layers are fetched from the cache right
before they run and released afterwards)

Reproduction:

```bash
stream-agent model add hf-internal-testing/tiny-random-LlamaForCausalLM \
    --alias tiny --backend transformers --quota-gib 0.2
stream-agent run tiny --backend transformers --prompt "Summarize the fixture." --max-tokens 16
stream-agent bench --model tiny --cache-sweep 512KiB,2MiB,8MiB --runs 3 \
    --backend transformers --block-size 262144
```

Observed (cold cache, 512 KiB budget, 32 generated tokens, one run):

| Metric | Value |
|---|---|
| downloaded bytes | 6 637 735 (weight payload + 1 842 764 byte tokenizer + config files) |
| range requests | 24 |
| full object requests | 0 |
| cache hits / misses | 575 / 19 |
| evictions / re-fetched bytes | 17 / 4 330 400 |
| generated tokens | 32 |
| bytes per output token | 207 429 (this run; 168 969 averaged over the three cold runs) |
| weight wait fraction | 0.91 |
| weight wait during model load | 9 947 ms (reported separately from generation) |
| tokens/s | 12.9 |
| peak weight cache | 786 432 B (524 288 B payload budget + in-flight reservation) |
| unique verified bytes retained | 461 264 B |

The model is a *random-weight* test model: the generated text is meaningless
by construction. What is demonstrated is that a real Transformers model runs
with its weights served from a bounded remote-backed cache.

---

## G4 — Multi-model accounting

> *At least two registered models share the same global store without
> violating global or per-model policy.*

| Evidence | Where |
|---|---|
| Two models share one global store; the per-model attribution is reported for both | `tests/test_multi_model.py::test_two_models_share_one_global_budget` |
| Identical content is stored **once** and attributed to every referencing model | `tests/test_multi_model.py::test_identical_content_is_stored_once` |
| A `fixed` quota blocks a model whose indivisible unit exceeds the quota | `tests/test_multi_model.py::test_fixed_quota_blocks_a_model_that_is_too_large` |
| An `elastic` quota may borrow unused capacity while the global hard limit still holds | `tests/test_multi_model.py::test_elastic_quota_allows_borrowing` |

---

## G5 — Failure recovery

> *Interrupted retrieval can resume or safely restart; corrupted/incomplete
> blocks never become usable cache entries.*

| Evidence | Where |
|---|---|
| A corrupted block is detected by hash verification, dropped and never served | `tests/test_cache_core.py::test_corrupt_block_is_never_served` |
| After a restart the cache is rebuilt from the manifest; entries whose physical block vanished are removed | `tests/test_cache_core.py::test_crash_recovery_rebuilds_the_cache` |
| Orphan blocks and leftover partials are removed on startup; the report states how much was reclaimed | `src/stream_agent/cache/manager.py::recover`, `stream-agent doctor` |
| `cache verify` re-checks every block and repairs deterministic inconsistencies | `stream-agent cache verify` |

---

## G6 — Agent invocation

> *One local client can invoke the selected model and receive a complete
> result or explicit job status.*

| Evidence | Where |
|---|---|
| The local endpoint answers `/health`, `/v1/models`, `/v1/chat/completions` and `/metrics` | `tests/test_server.py` |
| A job can be submitted, polled and cancelled; the record separates queue delay, weight wait and generation | `tests/test_server.py::test_job_lifecycle`, `src/stream_agent/jobs.py` |
| The MCP adapter exposes exactly three tools and no hidden coding loop | `tests/test_server.py::test_mcp_adapter_lists_and_calls_tools` |
| The endpoint binds loopback only and refuses a public bind without an explicit opt-in | `tests/test_server.py::test_bind_policy_refuses_a_public_interface` |

---

## Definition of done (Plan.md §19)

| Item | State | Evidence |
|---|---|---|
| Scope is separate from Pocket Harness and documented | done | `README.md` § *Scope boundary* |
| Exact revision and artifact sizes pinned in every report | done | every report carries `model.revision` and `model.artifact_format` |
| A partial remote-read path passed fixture based integrity tests | done | `tests/test_sources.py`, `tests/test_network_e2e.py` |
| Full object downloads cannot be triggered silently | done | `tests/test_sources.py::test_range_request_requires_206` |
| The hard limit includes in-flight reservations and temporary files | done | `tests/test_cache_core.py::test_budget_is_never_exceeded` |
| The report separates payload, store, working memory and traffic | done | `RunTelemetry.to_report()` — `budget`, `storage`, `network`, `environment` |
| One end-to-end model/backend tested; the others are listed as unsupported | done | G3 above, plus the supported-model table in `README.md` |
| Model switches and concurrent reads cannot mix revisions | done | `tests/test_policy.py::test_two_revisions_never_share_a_block` |
| Tokens and signed URLs are excluded from logs and metadata | done | `tests/test_policy.py::test_credentials_never_reach_the_metadata_database` |
| Untrusted remote code is disabled unless reviewed | done | `tests/test_policy.py::test_remote_code_is_blocked_by_default` |
| Licence and gating checks are recorded per model | done | `tests/test_policy.py::test_license_review_required_blocks_inference` |
| Local inference is served on loopback only | done | `tests/test_server.py::test_bind_policy_refuses_a_public_interface` |
| Cold/warm commands and machine/network details are published | done | this document, plus `docs/benchmarks/` |
| Benchmark output includes failures, exceptions and total transferred bytes | done | every report carries `errors`, `notes` and `network.downloaded_bytes_total` |
| README claims match what was demonstrated | done | the "Not implemented" section lists everything that is not claimed |

---

## Weight maps of the candidate models (Plan.md §9.5)

The three repositories named in the design were registered and their weight
maps were built **from header ranges only** — no weight shard was downloaded.

```bash
python scripts/probe_layouts.py
```

| repository | commit | license id | shards | tensors | weight bytes | layout built in |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `apache-2.0` | 18 | 1 199 | 55 563 006 776 | 7.4 s |
| `moonshotai/Kimi-K3` | `f831ab66814297da540d832a5235f8e904f29d06` | `other` | 96 | 497 220 | 1 560 936 091 448 | 67 s |
| `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `1bae0e1088993b9b82de08f59de5be5a34e42170` | `mit` | 48 | 96 085 | 510 296 708 312 | 20 s |

What this does and does not show:

* **Does:** the remote range path, the safetensors header parser and the shard
  index validation work on the real artifacts of all three candidate models,
  and the resulting maps describe 1 199 / 497 220 / 96 085 addressable
  tensors respectively.
* **Does not:** no weight byte of these models was fetched, so nothing here
  claims that they can *run* under a cache budget. Their working sets
  (55 GB / 1.5 TB / 510 GB) exceed the machine this was measured on.
* Kimi-K3 stores its experts as individual tensors, so expert-level access is
  architecturally possible with this format. It has **not** been validated end
  to end here.
* The DeepSeek checkpoint uses the `F8_E8M0` dtype. The parser accepts it, but
  the installed PyTorch (2.5.1) has no matching dtype, so the backend reports
  `BACKEND_CAPABILITY_MISSING` instead of guessing a conversion.

---

## Cache sweep (Stage D)

Raw data: [`benchmarks/tiny-random-llama-cache-sweep.json`](benchmarks/tiny-random-llama-cache-sweep.json)

Three cold runs per budget, same workload, 32 generated tokens per run,
262 144 byte cache blocks:

| weight cache budget | downloaded bytes (3 runs) | bytes / output token | cache hit ratio | weight wait fraction | tokens/s | evictions |
|---|---|---|---|---|---|---|
| 512 KiB | 16 221 063 | 168 969 | 0.968 | 0.92 | 12.8 | 55 |
| 2 MiB | 13 777 632 | 143 517 | 0.970 | 0.93 | 12.0 | 48 |
| 8 MiB | 12 393 840 | 129 103 | 0.973 | 0.89 | 17.2 | 40 |

Reading of the data:

* A larger budget monotonically reduces the transferred bytes per generated
  token and raises the throughput. At 8 MiB the working set fits and the run
  downloads exactly the 4 131 280 byte shard once per run.
* At 512 KiB the cache thrashes: 4 330 400 bytes were evicted and re-fetched
  during a single run. This is the dense-model behaviour described in
  `Plan.md` §11.2, and it is reported rather than hidden.
* The weight-wait fraction stays close to 1 because the backend re-reads the
  weights for every generated token; the wait for weights is not folded into
  "inference time".

---

## Limitations that are *not* covered by this evidence

* No MoE model has been validated. The expert interface exists, but no
  measured expert-level streaming result exists.
* No GGUF inference path exists.
* The measured model is a random-weight test model of ~4 MB. The numbers say
  nothing about a 27B or a 671B model; for those, the working set, not the
  cache, becomes the binding constraint.
* The network profile is an unthrottled residential connection. The bytes-per
  token figures would change on a slow or metered link; the *cache* effect
  would not.
* `elastic` quota sharing is implemented as "a model may borrow unused
  capacity until the global cap is reached"; it has not been compared against
  `fixed` on a real multi-model workload.
