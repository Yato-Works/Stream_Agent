"""Official Qwen2Moe Layer 0 Router Top-K Verification.

Loads Layer 0 using exact official Qwen2Moe classes (RMSNorm, SelfAttention, SparseMoeBlock),
executes the full forward pass on the prompt, and logs the EXACT top-4 experts selected
for each of the 5 tokens in the prompt.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer

from stream_agent.config import SourceConfig
from stream_agent.layout.model_map import build_layout
from stream_agent.sources.hf_hub import HuggingFaceSource

REPO = "Qwen/Qwen1.5-MoE-A2.7B-Chat"


async def fetch_exact_layer0():
    cache_dir = Path(".cache-qwen-moe")
    cache_dir.mkdir(exist_ok=True)

    source = HuggingFaceSource(SourceConfig())
    try:
        rev = await source.list_files(REPO, "main")
        layout = await build_layout(source, rev)
        cfg = AutoConfig.from_pretrained(REPO)
        tokenizer = AutoTokenizer.from_pretrained(REPO)

        async def load_param(name: str) -> torch.Tensor:
            bin_file = cache_dir / (name.replace(".", "_") + ".bin")
            if bin_file.is_file():
                data = bin_file.read_bytes()
            else:
                ref = layout.tensor(name)
                data = await source.fetch_range(REPO, rev.commit, ref.shard, ref.data_start, ref.data_end - 1)
                bin_file.write_bytes(data)
            ref = layout.tensor(name)
            dtype = torch.bfloat16 if ref.dtype == "BF16" else torch.float32
            return torch.frombuffer(bytearray(data), dtype=dtype).view(ref.shape)

        # 1. Instantiate Layer 0 with empty weights on CPU
        print("Instantiating official Qwen2MoeDecoderLayer(0)...")
        with torch.device("cpu"):
            layer = Qwen2MoeDecoderLayer(cfg, layer_idx=0).to(torch.bfloat16)

        # 2. Populate Layer 0 non-expert parameters
        prefix = "model.layers.0."
        layer.input_layernorm.weight.data.copy_(await load_param(f"{prefix}input_layernorm.weight"))
        layer.post_attention_layernorm.weight.data.copy_(await load_param(f"{prefix}post_attention_layernorm.weight"))

        layer.self_attn.q_proj.weight.data.copy_(await load_param(f"{prefix}self_attn.q_proj.weight"))
        layer.self_attn.q_proj.bias.data.copy_(await load_param(f"{prefix}self_attn.q_proj.bias"))
        layer.self_attn.k_proj.weight.data.copy_(await load_param(f"{prefix}self_attn.k_proj.weight"))
        layer.self_attn.k_proj.bias.data.copy_(await load_param(f"{prefix}self_attn.k_proj.bias"))
        layer.self_attn.v_proj.weight.data.copy_(await load_param(f"{prefix}self_attn.v_proj.weight"))
        layer.self_attn.v_proj.bias.data.copy_(await load_param(f"{prefix}self_attn.v_proj.bias"))
        layer.self_attn.o_proj.weight.data.copy_(await load_param(f"{prefix}self_attn.o_proj.weight"))

        layer.mlp.gate.weight.data.copy_(await load_param(f"{prefix}mlp.gate.weight"))
        layer.mlp.shared_expert_gate.weight.data.copy_(await load_param(f"{prefix}mlp.shared_expert_gate.weight"))
        layer.mlp.shared_expert.gate_proj.weight.data.copy_(await load_param(f"{prefix}mlp.shared_expert.gate_proj.weight"))
        layer.mlp.shared_expert.up_proj.weight.data.copy_(await load_param(f"{prefix}mlp.shared_expert.up_proj.weight"))
        layer.mlp.shared_expert.down_proj.weight.data.copy_(await load_param(f"{prefix}mlp.shared_expert.down_proj.weight"))

        # 3. Embed prompt tokens
        prompt = "Hello, my name is"
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        embed_ref = layout.tensor("model.embed_tokens.weight")
        row_size = 2048 * 2
        embed_rows = []
        for tid in input_ids:
            bin_f = cache_dir / f"embed_{tid}.bin"
            if bin_f.is_file():
                d = bin_f.read_bytes()
            else:
                s = embed_ref.data_start + tid * row_size
                e = s + row_size - 1
                d = await source.fetch_range(REPO, rev.commit, embed_ref.shard, s, e)
                bin_f.write_bytes(d)
            embed_rows.append(torch.frombuffer(bytearray(d), dtype=torch.bfloat16).view(1, 2048))
        hidden_states = torch.stack(embed_rows, dim=1)
        print(f"Prompt: {prompt!r} -> Tokens: {input_ids}")
        print(f"Prompt hidden states shape: {hidden_states.shape}")

        # 4. Run through Layer 0 Attention + Post-Attention Norm (Exact official math)
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeRotaryEmbedding
        rotary_emb = Qwen2MoeRotaryEmbedding(cfg).to("cpu")
        position_ids = torch.arange(len(input_ids), dtype=torch.long).unsqueeze(0)
        cos, sin = rotary_emb(hidden_states, position_ids)

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        attn_outputs = layer.self_attn(hidden_states, position_embeddings=(cos, sin))
        hidden_states = residual + attn_outputs[0]

        # Post attention layernorm -> this is the EXACT input into the MoE router!
        normed_for_moe = layer.post_attention_layernorm(hidden_states)

        # 5. Execute router forward across all prompt tokens
        router_logits, router_scores, router_indices = layer.mlp.gate(normed_for_moe)

        print("\n" + "=" * 65)
        print("OFFICIAL LAYER 0 ROUTER DECISIONS PER TOKEN (PREFILL STAGE):")
        all_unique_experts = set()
        for tok_idx, tid in enumerate(input_ids):
            token_str = tokenizer.decode([tid])
            selected = router_indices[tok_idx].tolist()
            scores = [round(s, 4) for s in router_scores[tok_idx].tolist()]
            all_unique_experts.update(selected)
            print(f"  Token {tok_idx} ({token_str!r:>10}): Top-4 Experts = {selected} (Scores: {scores})")

        print("=" * 65)
        print(f"Total prompt tokens: {len(input_ids)}")
        print(f"Union of selected experts across all 5 prompt tokens: {sorted(list(all_unique_experts))}")
        print(f"Total unique experts needed for prefill: {len(all_unique_experts)} of 60 experts")
        print(f"Prefill expert weight payload: {len(all_unique_experts) * 17.3:.1f} MB (vs {60 * 17.3:.1f} MB for all experts)")
        print(f"Actual Layer 0 Prefill Bandwidth Reduction: {100 * (1 - len(all_unique_experts)/60):.1f}%!")
        print("=" * 65)

    finally:
        await source.aclose()


if __name__ == "__main__":
    asyncio.run(fetch_exact_layer0())

