# MoE Feasibility Spike (2026-10-23)

## Candidate checkpoint
- allenai/OLMoE-1B-7B-0924 (64 experts, top-8, 16 layers, hidden 2048, intermediate 1024).
- Expert weights stored per-expert: model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight
  (192 tensors/layer x 16 layers = 3072 expert tensors + router gate).
- Total ~7 GB; individual expert tensor ~ (2048*1024*2 bytes for F16? actually bf16) ~4 MB each.

## Backend support
- transformers 5.20.2 has OlmoeForCausalLM; ExpertsInterface NOT present in this version.
- Stream Agent transformers backend reports supports_experts=False.
- No expert-selective weight loading in HF; loading a checkpoint loads ALL experts.

## Router expert-id extraction
- In OlmoeSparseMoeBlock, router (gate) produces top-k indices per token.
- Hook point: model.layers[i].mlp.gate returns (router_logits); topk gives expert ids.
- Stream Agent has no hook/callback surface to receive selected expert ids during forward.

## Source/Cache/Runtime connection
- Layout: ExpertPacking declared per tensor (tensor, dim, num_experts). For OLMoE, experts are separate tensors per id, not packed along a dim; ExpertPacking.slice_for assumes packed (dim,index,count). Mismatch: needs new packing model (per-expert tensors) or a virtual packed tensor.
- Provider.read_expert exists and maps ExpertKey -> byte range via layout.expert_element_range (not implemented for gguf/safetensors currently; tests only).
- Cache manager can bound retained blocks; expert tensors are ~4MB each, fits block granularity.

## Feasibility evaluation
- Full weight load unavoidable with current HF implementation: loading model loads all 64 experts per layer.
- Selective fetch requires custom modeling code: load only selected expert tensors into a sparse module, or monkeypatch expert forward.
- Risk: high. Requires (a) expert-capable backend, (b) custom MoE module, (c) router hook plumbing, (d) parity testing vs reference.

## Minimal prototype plan (if pursued)
1. Load model skeleton with expert weights missing (meta device) via accelerate.
2. Register forward hook on each layer gate to capture selected expert ids per batch.
3. On expert forward, fetch selected expert tensors via RemoteWeightProvider.read_expert (Range), cache-bound.
4. Inject weights into expert module parameters just-in-time; evict after use.
5. Record Range bytes/offsets and cache usage.
6. Compare outputs vs full-load reference (greedy, fixed seed).
7. Failure handling: if fetch fails, do not use partial weights; abort generation.

## Conclusion
- Not feasible as a small change. Keep MoE verification mechanism-only.
- Next: design expert-capable backend (custom module or transformers hook) before E2E.
