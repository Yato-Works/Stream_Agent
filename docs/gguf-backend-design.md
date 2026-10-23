# GGUF Backend Integration Design (2026-10-23)

## Goal
Connect the already-verified GGUF pieces (header parse + model-map build, tensor Range fetch,
whole-file fetch + inference + cache budget) into one integrated execution path, without
destabilising the Dense/Safetensors path.

## Current state
- `build_gguf_layout` builds a ModelLayout with one shard (the whole .gguf file) but does NOT
  populate `layout.tensors`. Hence `WeightProvider.read_tensor` cannot resolve GGUF tensors.
- Transformers can load a local GGUF via `from_pretrained(..., gguf_file=...)` (verified).
- Stream Agent has no GGUF-aware materialisation path; the transformers backend only handles
  safetensors tensor names.

## Two integration levels

### Level 1 (short-term, already evidenced): whole-file fetch into bounded cache
- Fetch the whole .gguf via `provider.read_file(shard, 0, size-1)` into the bounded cache.
- Write bytes to a temp file; call transformers `from_pretrained(..., gguf_file=...)`.
- Cache-budget compliance and inference already verified (gguf-cache-budget.json).
- Limitation: not tensor streaming; full file retained in cache.

### Level 2 (true streaming, not yet implemented)
- Populate `layout.tensors` with one TensorRef per GGUF tensor, computing byte range
  `data_offset + tensor.offset .. + size` (size requires GGML block math for quantized types).
- Backend materialises each parameter by `provider.read_tensor(TensorKey(name))` and dequantises
  GGUF payload to torch tensors (requires gguf/llama-cpp dequant kernels).
- This is the only path that achieves tensor-level Range fetch + cache selectivity.

## Required changes (Level 2)
1. `layout/gguf.py`: compute per-tensor byte size and populate `layout.tensors` (TensorRef).
   - Add GGML block-size table and numel->bytes conversion for types 0..28.
2. `backend/`: new `GgufBackend` (or `TransformersBackend` format branch) that:
   - builds skeleton on meta device,
   - materialises params from `provider.read_tensor`,
   - dequantises GGUF payloads to compute dtype.
3. Registry: register `gguf` backend without touching `transformers`/`fixture`.
4. Tests: one integrated run combining cold cache -> header -> tensor Range (byte-identical)
   -> inference -> budget -> full_object_requests==0.

## Risks
- Dequantisation correctness: must match transformers output bit-for-bit for parity.
- Quantized tensor slicing not supported by current ExpertPacking/TensorRef slicing; GGUF
  tensors are fetched whole.
- Memory: dequantised params expand to compute dtype; working set may exceed budget.

## Decision
Keep Dense/Safetensors as the stable baseline. Implement Level 1 behind a dedicated backend
first (no dense changes), then attempt Level 2 as a separate spike.
