# Stream Agent

**Storage-budgeted on-demand model runtime.**

Stream Agent lets you set an explicit local storage budget for model weights
and then use compatible open-weight models through on-demand weight retrieval
and a shared, bounded cache — without keeping every model's complete weight
files on your SSD.

The design document that governs this implementation is [`Plan.md`](Plan.md).
This README describes what the implementation actually does today, and what
it explicitly does **not** claim.

---

## Status

This is an experimental implementation of the `v0.1-design` specification.
The following is implemented and covered by the test-suite:

| Area | Component | State |
|---|---|---|
| C0 | CLI and configuration loader | implemented |
| C1 | Model catalog / revision resolver | implemented |
| C2 | Remote source adapter (Hugging Face Hub, incl. the Xet/CDN range path) | implemented |
| C3 | Model layout / weight map (safetensors, GGUF metadata) | implemented |
| C4 | `WeightProvider` / block reader | implemented |
| C5 | Global cache manager (budget, reservations, eviction, single-flight, crash recovery) | implemented |
| C6 | Prefetch scheduler | implemented (bounded, no expert prediction) |
| C7 | Inference backend adapter | `transformers` (lazy + eager) and `fixture` |
| C8 | Local model server / worker controller | implemented (loopback only) |
| C9 | Telemetry and local report | implemented |
| §12.3 | Asynchronous job API | implemented |
| §12.4 | MCP adapter | implemented (stdio, three tools) |

Not implemented (and therefore not claimed):

* MoE expert-level streaming for a real MoE model — the backend interface
  supports declared expert packing, but no MoE model has been validated yet.
* GGUF inference — the metadata parser exists, the remote-backed reader path
  does not.
* Learned prefetch prediction, LRU-K/2Q/LFU eviction, elastic quota sharing
  beyond the simple "borrow under the global cap" behaviour.
* Token-level streaming from the local endpoint (the endpoint refuses
  `stream: true` with a typed error instead of pretending).
* Commercial-use or redistribution of any model's weights.

---

## Install

```bash
python -m pip install -e ".[server,torch,hub,dev]"
```

The core runtime only needs `httpx`. `fastapi`/`uvicorn` are needed for the
local endpoint, `torch`/`transformers` for the real inference backend.

## Quick start

```bash
# 1. Check the environment, the budget and the registered models
stream-agent doctor

# 2. Register a pinned revision. This never downloads weight bytes.
stream-agent model add hf-internal-testing/tiny-random-LlamaForCausalLM \
    --alias tiny --backend transformers --quota-gib 0.2

# 3. Look at the storage accounting
stream-agent cache status

# 4. Run one bounded local generation
stream-agent run tiny --prompt "Summarize the fixture." --max-tokens 32

# 5. Sweep the cache budget and record the metrics
stream-agent bench --model tiny --cache-sweep 512KiB,2MiB,8MiB --runs 2 \
    --block-size 262144

# 6. Serve the model on loopback only
stream-agent serve --model tiny --port 8789
```

## Scope boundary

Stream Agent is a **model execution endpoint**, not an agent. It discovers a
model revision, retrieves the weight data a backend asks for, enforces the
storage budget and reports the cost of that choice.

It is a **separate project from Pocket Harness** and deliberately does not
implement a request/response repair proxy, context compression, tool
selection, tool-call repair, a loop guard, an execution ledger or a
coding-task benchmark. It may expose a local inference endpoint that an
existing agent calls, but it does not try to make that agent's history or
tool behaviour more reliable.

## Supported models

| repository | format | weight map | end-to-end generation | notes |
|---|---|---|---|---|
| `hf-internal-testing/tiny-random-LlamaForCausalLM` | safetensors | yes | yes (Transformers backend) | random-weight test model; used for the benchmarks |
| `Qwen/Qwen3.8-27B` | safetensors | yes (1 199 tensors) | no | 55.6 GB of weights; the working set does not fit this machine |
| `moonshotai/Kimi-K3` | safetensors | yes (497 220 tensors) | no | 1.56 TB of weights; licence id is `other`, review required |
| `deepseek-ai/DeepSeek-V4.1-Flash-1017` | safetensors | yes (96 085 tensors) | no | 510 GB of weights; uses the `F8_E8M0` dtype |

"weight map" means the tensor map was built from header ranges only, without
downloading a weight shard. "end-to-end generation" means a real Transformers
model produced tokens through the bounded remote-backed cache.

Anything else is **unsupported**: the runtime fails with
`MODEL_FORMAT_UNSUPPORTED`, `BACKEND_CAPABILITY_MISSING` or
`RANGE_UNSUPPORTED` instead of downloading the whole artifact.

## The two limits

The configuration distinguishes two budgets and the CLI never calls either of
them "total storage":

* `weight_cache_limit_gib` — hard limit for retained and in-flight **weight
  payload** blocks.
* `managed_store_limit_gib` — hard limit for the **whole managed directory**
  (payload + metadata + logs + temporary files).

If your 20 GiB is everything you have, set the managed-store cap and leave
headroom for metadata and staging. Metadata files (config, tokenizer, shard
index) are stored separately from the weight payload and are reported
separately.

## What a small cache does and does not mean

A 3 GiB cache does **not** turn a 70 GB model into a 3 GB model. It means that
at most 3 GiB of *downloaded weight payload* is kept on disk; everything else
is fetched again when the backend asks for it. Working memory (RAM/VRAM) is a
separate resource, and the runtime reports bytes-per-generated-token so that
the cost of a small cache is visible rather than hidden.

## Architecture

```
CLI / local client / agent wrapper
        |
        v
Local Stream Agent API  --->  Session and Job Controller
        |                              |
        v                              v
Inference Backend Adapter ----> WeightProvider
                                     |
                       Model Layout / Weight Map
                                     |
                             Remote Source Adapter  --->  Hub / Xet
                                     |
                        Global Cache Manager
                                     |
                    Content-addressed Weight Block Store
                                     |
                          Metadata DB + Usage Ledger
```

The cache manager owns the central contract:

* the persistent weight cache never exceeds the configured hard limit,
  including in-flight reservations (capacity is reserved *before* a byte is
  written);
* only verified data is served; a corrupted block is dropped, never returned;
* revisions are pinned — weight blocks from different commits are never mixed;
* concurrent requests for the same immutable block share one transfer
  (single-flight);
* pinned blocks are never evicted, and a pinned working set that cannot fit
  the budget produces `WORKING_SET_EXCEEDS_BUDGET` instead of a deadlock.

## Error contract

Every failure carries a stable code (see `Plan.md` §15): `SOURCE_AUTH`,
`SOURCE_FORBIDDEN`, `SOURCE_NOT_FOUND`, `SOURCE_RATE_LIMITED`,
`SOURCE_UNAVAILABLE`, `RANGE_UNSUPPORTED`, `RANGE_MISMATCH`,
`INTEGRITY_FAILURE`, `MODEL_FORMAT_UNSUPPORTED`,
`BACKEND_CAPABILITY_MISSING`, `CACHE_BUDGET_EXCEEDED`,
`WORKING_SET_EXCEEDS_BUDGET`, `BACKEND_OOM`, `SESSION_CANCELLED`,
`REMOTE_CODE_BLOCKED`, `MODEL_LICENSE_REVIEW_REQUIRED`.

```bash
stream-agent --json run tiny --prompt "hi"   # errors are printed as [CODE] message
```

## Local endpoint

`stream-agent serve --model <alias>` exposes, on loopback only:

| Route | Purpose |
|---|---|
| `GET /health` | process and backend health, no secrets |
| `GET /v1/models` | registered models and the implemented capability fields |
| `POST /v1/chat/completions` | one bounded inference request |
| `GET /metrics` | local runtime metrics |
| `POST /v1/jobs`, `GET /v1/jobs/{id}`, `POST /v1/jobs/{id}/cancel` | slow background work |

The endpoint is an *OpenAI-compatible subset*, not a full clone: only the
listed request fields are honoured and unimplemented options return a clear
error.

## Agent integration

The MCP adapter (`stream-agent mcp --model <alias>`) exposes exactly three
tools — `stream_agent_submit_review`, `stream_agent_job_status`,
`stream_agent_cancel` — over stdio. It does not implement a coding loop, a
planner or a tool executor; the parent agent keeps that responsibility, and
the latency expectation is stated in the tool description.

## Security and licence posture

* `allow_remote_code = false` by default; a repository that needs custom code
  is blocked with `REMOTE_CODE_BLOCKED` until it is explicitly reviewed.
* Tokens are read from an environment variable or a credential store. They are
  never written to the configuration, to a report or to a log, and signed URLs
  are never persisted.
* The licence identifier of a model is recorded per revision. It is an
  indicator, not a legal conclusion; `license_review = "required"` blocks
  unattended inference for that model.
* Stream Agent never mirrors or re-hosts model weights.

## Tests

```bash
python -m pytest -q            # offline suite
python -m pytest -q -m network # adds the live repository experiment
```

The live test performs the experiment proposed in `Plan.md` Appendix C: pin a
revision, read only the header, fetch a single tensor range, compare it byte
for byte with a locally downloaded reference, and assert that the complete
artifact was never pulled into the cache.

`python scripts/probe_layouts.py` builds the weight map of the three
candidate repositories named in the design (`Qwen/Qwen3.8-27B`,
`moonshotai/Kimi-K3`, `deepseek-ai/DeepSeek-V4.1-Flash-1017`) from header
ranges only — no weight shard is downloaded. The measured results are in
[`docs/EVIDENCE.md`](docs/EVIDENCE.md).

## Evidence

[`docs/EVIDENCE.md`](docs/EVIDENCE.md) maps the acceptance gates G0–G6 of the
design onto the tests and the recorded runs that back them, and states the
limitations that the evidence does *not* cover. The raw benchmark output is
in [`docs/benchmarks/`](docs/benchmarks/).

## Prior art

Streaming model weights from storage is not a new idea. [PULSAR](https://github.com/giannisanni/pulsar),
[Edge0](https://github.com/Edge0-AI/Edge0) and the
[Hugging Face Xet protocol](https://huggingface.co/docs/xet/en/download-protocol)
all address parts of this problem. Stream Agent's differentiation hypothesis
is the combination of a *remote* repository as the backing source, a *global
hard disk budget shared by several registered models*, backend-independent
storage orchestration, an agent-friendly local endpoint, and first-class
measurement of downloaded bytes and cache churn. That is a hypothesis, not a
proof of novelty.

## Licence

The Stream Agent source is Apache-2.0. Model weights keep their own terms:
see `Plan.md` §14.4.
