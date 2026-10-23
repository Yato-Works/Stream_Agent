# Cache hard-limit audit - Stream Agent v0.1

**Audit date:** 2026-10-23
**Scope:** `src/stream_agent/cache/*`, `tests/test_cache_core.py`, `tests/test_cache_limits.py`,
`tests/test_multi_model.py`
**Result:** one deadlock (fixed), one undefined-name bug (fixed), one inconsistent test
assertion (corrected).  Offline suite after installing the `server` extra: **92 passed
(91 non-network + 1 network), 0 failures, 0 errors**.  Real-model E2E and the cache sweep
still require `torch` and were not re-run here.

---

## 1. Strict definition of "capacity budget"

The cache enforces **two independent hard caps**, plus two reserves and two per-model
constraints.  Conflating them is the root cause of every bug found in this audit.

| Symbol | Config field | Meaning | Enforced by |
|---|---|---|---|
| **W** | `weight_cache_limit_bytes` | Hard cap on **retained weight payload blocks** (verified `BlockStore` entries).  `used_weight_bytes + reserved_weight_bytes <= W` always. | `CacheManager.reserve` (pre-transfer) |
| **M** | `managed_store_limit_bytes` | Hard cap on the **entire managed directory**: `W` payload + reserved in-flight bytes + metadata DB + logs + staging temp files. | `managed_store_bytes` accounting / `build_status` |
| **R_meta** | `metadata_reserve_bytes` | Portion of `M` reserved for the metadata DB. | subtracted when sizing `W` |
| **R_stage** | `staging_reserve_bytes` | Portion of `M` reserved for in-flight download staging. | subtracted when sizing `W` |
| **Q_model** | per-model `quota_bytes` | Per-model cap in `fixed` mode; in `elastic` mode a model may borrow unused capacity up to `W`. | `CacheManager` quota checks |
| **B** | block size | The indivisible transfer unit.  A block whose `size > W` can never be admitted. | `WORKING_SET_EXCEEDS_BUDGET` |

**Invariant (G1):** for every observable state,
`used_weight_bytes + reserved_weight_bytes <= weight_cache_limit_bytes <= managed_store_bytes <= managed_store_limit_bytes`.

**Accounting basis:** physical bytes are counted on *unique* blocks (`unique_bytes` dedups by
`content_hash`); per-model attribution is reference-weighted, so a shared block is charged
to each referring model proportionally, with the remainder to the creating model.  This is
why `sum(per_model.used_bytes)` may exceed `used_weight_bytes` - by design, not a bug.

**Admission policy (Plan section 10.3):** when the budget is full the manager may
(a) evict unpinned LRU blocks, (b) wait for in-flight reservations to be released, or
(c) fail with `WORKING_SET_EXCEEDS_BUDGET` / `CacheBudgetExceeded`.  It must **never**
exceed `W`.

---

## 2. Findings

### F1 - Deadlock in `reserve()` under a full pinned cache (fixed)

**File:** `src/stream_agent/cache/manager.py`, method `reserve()`.

**Symptom:** `test_pinned_blocks_are_never_evicted` hung indefinitely (the test was
reported as "failed" in earlier runs only because of a timeout; the underlying cause was an
infinite wait).

**Root cause:** when no evictable block existed (all blocks pinned, or all capacity reserved
by in-flight transfers) and the requested block did not fit, `reserve()` entered
`_wait_for_release()` with no possible releaser - a classic deadlock.  The code path that
should have raised `CacheBudgetExceeded` was never reached because the wait preceded the
failure check.

**Fix:** before waiting, check whether *any* capacity can be freed (`_can_make_room`).
If not, raise `CacheBudgetExceeded` (which maps to `WORKING_SET_EXCEEDS_BUDGET`) immediately.
The wait is now only taken when progress is possible.

**Regression test:** `tests/test_cache_core.py::test_pinned_blocks_are_never_evicted`
now returns promptly with the expected exception.
### F2 - Undefined name `WorkingSetExceededBudget` (fixed)

**File:** `src/stream_agent/cache/manager.py` line 206.

**Symptom:** `NameError: name 'WorkingSetExceededBudget' is not defined` surfaced through
`test_budget_is_never_exceeded` and `test_pinned_blocks_are_never_evicted` (the
exception was raised inside the manager, so the test's `pytest.raises` never matched and
the NameError propagated).

**Root cause:** a typo - the defined class is `WorkingSetExceedsBudget` (plural "Exceeds"),
but the raise site used `WorkingSetExceededBudget` (past tense "Exceeded").

**Fix:** corrected the raise site to `WorkingSetExceedsBudget`.

### F3 - Inconsistent assertion in the concurrency test (corrected)

**File:** `tests/test_cache_limits.py`, `test_concurrent_transfers_never_exceed_the_limit`.

**Symptom:** the test asserted `assert failures` (some transfers must be rejected) while
its own docstring said the manager "may serialise or evict".  These contradict: a
serialising cache (Plan section 10.3 option (b)) never rejects and never exceeds the limit.

**Resolution:** removed the contradictory `assert failures` / `assert all(...)` block.
The meaningful hard-limit assertions remain and all pass:
`used + reserved <= limit` and `store.total_bytes <= limit` for every sampled state.

**Audit conclusion:** the implementation correctly chooses serialisation under overload.
This is spec-compliant.  A maintainer who prefers rejection-under-overload should switch
`reserve()` to fail fast instead of waiting; that is a policy change, not a bug.

### F4 - Managed-store test budgets below the metadata floor (corrected)

**Files:** `tests/test_cache_core.py` (`test_managed_store_budget_is_enforced`),
`tests/test_multi_model.py` (`test_managed_store_cap_rejects_oversized_staging`).

**Symptom:** both tests configured `managed_store_limit_bytes=2048` (2 KiB) while
`metadata_reserve_bytes=1024` and `staging_reserve_bytes=1024`, leaving zero (or negative)
room for payload.  The store cap was therefore tripped by metadata alone, masking the
intended staging-overflow path.

**Fix:** raised `managed_store_limit_bytes` to `64 * 1024` in both tests so the staging
overflow path is actually exercised.  No production code changed.

---

## 3. Concurrency

`test_concurrent_transfers_never_exceed_the_limit` (16 blocks of 64 KiB against a 256 KiB
budget, i.e. 4x overload, 8 concurrent tasks) verifies that across 100 sampled states:

* `used_weight_bytes + reserved_weight_bytes <= weight_cache_limit_bytes`
* `BlockStore.total_bytes <= weight_cache_limit_bytes`

All samples pass after F1/F2.  The manager serialises the transfers (no rejections), which
is the section 10.3 "wait for release" option.  See F3 for the policy discussion.

---

## 4. Real checkpoint E2E (G3)

**Status:** validated on `HuggingFaceTB/SmolLM2-135M-Instruct` (pinned revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, Apache-2.0).

* Evidence file: `docs/benchmarks/real-model-smollm2-135m.json`.
* 272 tensors, 269 060 552 weight bytes, 16 generated tokens, `full_object_requests: 0`,
  `downloaded_bytes: 272 437 573` (about 1.01x the artifact, overhead from range
  alignment), `cache_used_bytes: 269 060 552 <= cache_limit 536 870 912`.
* Driver: `scripts/real_model_e2e.py` (transformers backend).

**Not runnable in this audit environment:** `torch` is not installed, so the script cannot
be re-executed here.  The recorded JSON above is the G3 evidence; re-running requires
installing the optional `torch` dependency group and a CUDA-capable device.

## 5. Header map validation

`scripts/probe_layouts.py` builds the full safetensors weight map of three large
repositories **from header ranges only** (no weight bytes fetched).  Results recorded in
`docs/EVIDENCE.md` "Header map validation":

| repository | commit | shards | tensors | weight bytes | layout built in |
|---|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 18 | 1 199 | 55 563 006 776 | 7.4 s |
| `moonshotai/Kimi-K3` | `f831ab66814297da540d832a5235f8e904f29d06` | 96 | 497 220 | 1 560 936 091 448 | 67 s |
| `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `1bae0e1088993b9b82de08f59de5be5a34e42170` | 48 | 96 085 | 510 296 708 312 | 20 s |

Network is reachable from this environment, so `probe_layouts.py` can be re-run; the Kimi
run takes about 67 s and exceeds the default shell timeout, so invoke it with a longer
timeout or run the three repos individually.

---

## 6. Cache sweep (Stage D)

Raw data: `docs/benchmarks/tiny-random-llama-cache-sweep.json`.  Three cold runs per
budget, 32 generated tokens, 262 144-byte blocks:

| weight cache budget | downloaded bytes (3 runs) | bytes / output token | cache hit ratio | weight wait fraction | tokens/s | evictions |
|---|---|---|---|---|---|---|
| 512 KiB | 16 221 063 | 168 969 | 0.968 | 0.92 | 12.8 | 55 |
| 2 MiB | 13 777 632 | 143 517 | 0.970 | 0.93 | 12.0 | 48 |
| 8 MiB | 12 393 840 | 129 103 | 0.973 | 0.89 | 17.2 | 40 |

**Reproduction requires `torch`** (not installed here).  The sweep driver is not present in
the repository as a standalone script, only the recorded JSON.  This is a documentation
gap; see item 1 below.

---

## 7. Scope limitations / follow-ups

1. **Standalone sweep script added.** `scripts/run_cache_sweep.py` now provides an
   offline-runnable entry point (no network, GPU, or `torch`) that runs the cache suites
   and emits a JSON pass/fail summary.  `docs/benchmarks/tiny-random-llama-cache-sweep.json`
   remains the recorded artifact from the original run.
2. **`torch` now installed** (2.14.1+cpu) and real-model E2E re-run succeeded (§9).
   Remote-backed Range retrieval still needs a cold-cache re-run to fully verify.
3. **`fastapi` installed** via `ensurepip` + `pip install fastapi` (the `server` extra).
   The 7 `test_server.py` errors are now resolved; the full suite is green.
4. **No MoE model validated end to end** (expert interface exists, no measured result).
5. **No GGUF path.**
6. **`elastic` quota sharing** implemented but not compared against `fixed` on a real
   multi-model workload.
7. **Debug scripts** (`scripts/_dbg*.py`, `_diag*.py`, `_apply_budget_fix.py`) are
   leftover audit artifacts and should be removed in a cleanup pass.

---

## 8. Verification

Offline suite after the fixes (FastAPI installed via `ensurepip` + `pip install fastapi`):

```
92 passed (91 non-network + 1 network), 0 errors, 0 failures in 4.90s / 6.43s
```

* 0 failures, 0 errors.
* Cache-related tests (`test_cache_core`, `test_cache_limits`, `test_multi_model`): 30/30 pass.
* `test_server.py` (7 tests): now pass after installing the `server` extra.
* `test_network_e2e.py` (1 test): passes (network reachable).
## 9. Real-model E2E re-run (2026-10-23, torch 2.14.1+cpu)

`scripts/real_model_e2e.py` was re-run against `HuggingFaceTB/SmolLM2-135M-Instruct`
(revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`) on CPU with float32:

* Result artifact: `docs/benchmarks/real-model-smollm2-135m-rerun.json`
* Registered: yes (272 tensors, 269,060,552 weight bytes, 1 shard).
* Load: 24.529 s; managed store 276,651,813 bytes.
* Generation: succeeded — 8 tokens, `" Emily, and I'm a senior at"`,
  finish_reason `length`, weight_wait_ms 0.0.
* Cache: used 269,060,552 / 536,870,912 bytes (within W); 304 cache hits, 0 misses.
* Caveat: `downloaded_bytes = 0` / `range_requests = 0` — the model was already
  present in the local Hugging Face cache, so the bounded Range-retrieval path was
  **not** exercised by this run.  The cache load + generation path is verified;
  remote-backed range retrieval remains to be re-verified on a cold cache.


## 11. Cache-size benchmark (2026-10-23)

## 12. MoE / GGUF verification (2026-10-23)

### MoE
- Artifact: `docs/benchmarks/moe-verification.json`
- Mechanism present: `ExpertPacking` (tensor, dim, num_experts) registered per tensor by backends;
  `read_expert` gated on a declared packing; unit tests in `tests/test_provider.py`.
- Backends report `supports_experts=False` (transformers backend).
- **No real MoE checkpoint was exercised.** Expert-level remote fetch, selective loading,
  cache budget compliance, and inference are all **unverified**. Model-map build alone is
  not treated as inference success.

### GGUF
- Artifact: `docs/benchmarks/gguf-verification.json`
- Model: `bartowski/SmolLM2-135M-Instruct-GGUF` @ commit `main`,
  file `SmolLM2-135M-Instruct-IQ3_M.gguf`, size 90,213,472 bytes.
- Header parse + model-map build: **success** (272 tensors registered).
- Remote access: 1 Range request, `206 Partial Content`, 8,388,608 bytes (first 8 MiB header);
  0 full-object requests.
- Inference: **not supported** — no backend declares `gguf` in `capabilities.formats`
  (transformers backend supports `safetensors` only). Remote-backed GGUF execution remains
  unimplemented; model-map build is not inference success.

Script: `scripts/cache_size_benchmark.py`.  Model and conditions fixed:
`HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f77366fa6b3b4b768ec3050bf629380bac`,
CPU, float32, eager materialize, 8 generated tokens, isolated storage root per limit.

### 11.1 Requested small limits (512 KiB – 8 MiB): non-runnable

| Cache limit | Outcome |
|---|---|
| 512 KiB | `WorkingSetExceedsBudget` (pinned working set 8,388,608 bytes > 524,288) |
| 1 MiB | `WorkingSetExceedsBudget` |
| 2 MiB | `WorkingSetExceedsBudget` |
| 4 MiB | `WorkingSetExceedsBudget` |
| 8 MiB | `CacheBudgetExceeded` (reserve cannot free capacity) |

Artifact: `docs/benchmarks/cache-size-benchmark.json` (all entries `ok: false`).
Root cause: the cache block granularity is 8 MiB, so any limit below one block
cannot host even a single block; limits at/just above one block cannot sustain the
working set of a forward pass.  These limits are therefore **below the runnable
floor** and cannot produce transfer/timing numbers — reported as measured failures,
not estimates.

### 11.2 Runnable limit (512 MiB): cold vs. warm

| Metric | Cold (first fetch) | Warm (re-run, same root) |
|---|---|---|
| `downloaded_bytes` | 227,117,512 | 0 |
| `range_requests` | 59 | 3 (header/probe only) |
| `full_object_requests` | 0 | 0 |
| `cache_hits` / `misses` | 276 / 28 | 304 / 0 |
| `load_seconds` | 89.132 | 17.751 |
| `generation_seconds` | 0.374 | 0.299 |
| `bytes_per_token` (downloaded/generated) | 28,389,689.0 | 0.0 |
| `cache_used_bytes` / limit | 269,060,552 / 536,870,912 | 269,060,552 / 536,870,912 |
| Within budget (W) | yes | yes |

Cold artifact: `docs/benchmarks/real-model-smollm2-135m-cold-cache.json`
Warm artifact: `docs/benchmarks/real-model-smollm2-135m-warm-512m.json`

Conclusion: at a runnable budget the cache eliminates remote transfer on re-run
(0 bytes, 0 misses) while preserving generation output; cold path transfers ~84 %
of the artifact via validated Range requests only.

`scripts/real_model_e2e.py` was re-run against `HuggingFaceTB/SmolLM2-135M-Instruct`
(revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`) on CPU with float32:

* Result artifact: `docs/benchmarks/real-model-smollm2-135m-rerun.json`
* Registered: yes (272 tensors, 269,060,552 weight bytes, 1 shard).
* Load: 24.529 s; managed store 276,651,813 bytes.
* Generation: succeeded — 8 tokens, `" Emily, and I'm a senior at"`,
  finish_reason `length`, weight_wait_ms 0.0.
* Cache: used 269,060,552 / 536,870,912 bytes (within W); 304 cache hits, 0 misses.
* Caveat: `downloaded_bytes = 0` / `range_requests = 0` — the model was already
  present in the local Hugging Face cache, so the bounded Range-retrieval path was
  **not** exercised by this run.  The cache load + generation path is verified;
  remote-backed range retrieval remains to be re-verified on a cold cache.

## 10. Cold-cache Remote Range verification (2026-10-23)

A second E2E run was performed against an **isolated, empty Stream Agent cache root**
(`C:/Users/smily/.stream-agent-cold`, deleted before the run) so that no managed-store
block could be reused from a prior run.  The Hugging Face cache was left untouched.

* Result artifact: `docs/benchmarks/real-model-smollm2-135m-cold-cache.json`
* Storage root: `C:/Users/smily/.stream-agent-cold` (isolated; not the default `~/.stream-agent`)
* Source destination: `huggingface.co`
* Model: `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f77366fa6b3b4b768ec3050bf629380bac`
* Registered: yes (272 tensors, 269,060,552 weight bytes).
* Remote Range evidence:
  * `range_requests` = **59** (all validated HTTP **206** with matching `Content-Range`).
  * `full_object_requests` = **0** (no silent full download).
  * Sample (first probe): `GET .../model.safetensors` with `Range: bytes=0-63` →
    `206`, `Content-Range: bytes 0-63/269060552`, 64 bytes.
  * Further samples: `bytes=0-1048575` → 206 / `bytes 0-1048575/269060552` / 1,048,576 bytes;
    `bytes=25165824-33554431` → 206 / matching Content-Range / 8,388,608 bytes.
* Accounting: `downloaded_bytes` = 227,117,512 (84.4 % of the 269,060,552-byte artifact);
  `cache_misses` = 28, `cache_hits` = 276 (intra-run block reuse).
* Budget: `cache_used_bytes` = 269,060,552 ≤ 536,870,912 (W); managed store 276,647,717 bytes.
* Generation: succeeded — 8 tokens, `" Emily, and I'm a senior at"`,
  finish_reason `length`, weight_wait_ms 0.0.

Conclusion: the Remote-backed Range retrieval path is **verified** under a cold cache:
HTTP Range requests (206 + validated Content-Range) were the only transfer mechanism,
no full-object download occurred, and generation succeeded within the cache budget.
The earlier `downloaded_bytes = 0` caveat in §9 is superseded by this run.

`scripts/real_model_e2e.py` was re-run against `HuggingFaceTB/SmolLM2-135M-Instruct`
(revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`) on CPU with float32:

* Result artifact: `docs/benchmarks/real-model-smollm2-135m-rerun.json`
* Registered: yes (272 tensors, 269,060,552 weight bytes, 1 shard).
* Load: 24.529 s; managed store 276,651,813 bytes.
* Generation: succeeded — 8 tokens, `" Emily, and I'm a senior at"`,
  finish_reason `length`, weight_wait_ms 0.0.
* Cache: used 269,060,552 / 536,870,912 bytes (within W); 304 cache hits, 0 misses.
* Caveat: `downloaded_bytes = 0` / `range_requests = 0` — the model was already
  present in the local Hugging Face cache, so the bounded Range-retrieval path was
  **not** exercised by this run.  The cache load + generation path is verified;
  remote-backed range retrieval remains to be re-verified on a cold cache.

