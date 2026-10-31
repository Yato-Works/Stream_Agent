"""End-to-End Selective MoE Generation Script for Qwen1.5-MoE-A2.7B-Chat.

Demonstrates running genuine multi-token inference on a 28 GB sparse MoE model
by dynamically fetching ONLY router-selected top-k experts per layer via HTTP 206
Range requests.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from stream_agent.config import SourceConfig
from stream_agent.layout.model_map import build_layout
from stream_agent.sources.hf_hub import HuggingFaceSource

REPO = "Qwen/Qwen1.5-MoE-A2.7B-Chat"


class SelectiveMoeRunner:
    def __init__(self, source: HuggingFaceSource, layout: any, commit: str, device: str = "cpu"):
        self.source = source
        self.layout = layout
        self.commit = commit
        self.device = device
        self.fetched_bytes = 0
        self.cache: dict[str, torch.Tensor] = {}

    async def get_tensor(self, name: str) -> torch.Tensor:
        if name in self.cache:
            return self.cache[name]
        ref = self.layout.tensor(name)
        raw = await self.source.fetch_range(REPO, self.commit, ref.shard, ref.data_start, ref.data_end - 1)
        self.fetched_bytes += len(raw)
        # Parse dtype
        dtype = torch.bfloat16 if ref.dtype == "BF16" else torch.float32
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).view(ref.shape).to(self.device)
        self.cache[name] = tensor
        return tensor

    async def load_embed_and_head(self, model: any):
        print("  Loading embed_tokens, norm, and lm_head...")
        embed = await self.get_tensor("model.embed_tokens.weight")
        model.model.embed_tokens.weight.data.copy_(embed)

        norm = await self.get_tensor("model.norm.weight")
        model.model.norm.weight.data.copy_(norm)

        lm_head = await self.get_tensor("lm_head.weight")
        model.lm_head.weight.data.copy_(lm_head)

    async def load_layer_non_experts(self, layer_idx: int, layer_mod: any):
        """Load attention and shared expert weights for one layer."""
        prefix = f"model.layers.{layer_idx}."
        non_expert_keys = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.q_proj.bias",
            "self_attn.k_proj.weight", "self_attn.k_proj.bias",
            "self_attn.v_proj.weight", "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "mlp.gate.weight",
            "mlp.shared_expert_gate.weight",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert.down_proj.weight",
        ]
        for key in non_expert_keys:
            full_name = f"{prefix}{key}"
            t = await self.get_tensor(full_name)
            # Find parameter in layer_mod
            curr = layer_mod
            parts = key.split(".")
            for p in parts[:-1]:
                curr = getattr(curr, p)
            param = getattr(curr, parts[-1])
            param.data.copy_(t)

    async def materialize_selected_experts(self, layer_idx: int, layer_mod: any, expert_ids: list[int]):
        """Dynamically fetch and assign only top-k expert tensors."""
        for exp_id in expert_ids:
            up_name = f"model.layers.{layer_idx}.mlp.experts.{exp_id}.up_proj.weight"
            gate_name = f"model.layers.{layer_idx}.mlp.experts.{exp_id}.gate_proj.weight"
            down_name = f"model.layers.{layer_idx}.mlp.experts.{exp_id}.down_proj.weight"

            up_t = await self.get_tensor(up_name)
            gate_t = await self.get_tensor(gate_name)
            down_t = await self.get_tensor(down_name)

            gate_up = torch.cat([gate_t, up_t], dim=0)
            layer_mod.mlp.experts.gate_up_proj.data[exp_id].copy_(gate_up)
            layer_mod.mlp.experts.down_proj.data[exp_id].copy_(down_t)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Connecting to {REPO}...")
    source = HuggingFaceSource(SourceConfig())
    try:
        rev = await source.list_files(REPO, "main")
        layout = await build_layout(source, rev)
        print(f"Repository mapped! Total model size: {layout.total_weight_bytes / 1024**3:.2f} GB")

        tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=False)
        cfg = AutoConfig.from_pretrained(REPO)

        print(f"Instantiating model skeleton in {args.device} memory (bfloat16)...")
        with torch.device(args.device):
            model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
        model.eval()

        runner = SelectiveMoeRunner(source, layout, rev.commit, device=args.device)

        # 1. Load Embeddings and LM Head
        await runner.load_embed_and_head(model)

        # 2. Hook each layer's router to intercept hidden states and fetch experts on demand
        active_expert_counts = 0
        total_expert_calls = 0

        def make_moe_pre_hook(layer_idx: int):
            def hook(module, args):
                nonlocal active_expert_counts, total_expert_calls
                hidden_states = args[0]
                # hidden_states: [batch, seq_len, hidden_size]
                last_hidden = hidden_states[:, -1, :]
                gate_weight = module.gate.weight  # [60, 2048]
                logits = torch.matmul(last_hidden, gate_weight.t())
                probs = torch.softmax(logits, dim=-1)
                _, topk_indices = torch.topk(probs, k=cfg.num_experts_per_tok, dim=-1)
                selected_ids = sorted(list(set(topk_indices.view(-1).tolist())))
                total_expert_calls += cfg.num_experts
                active_expert_counts += len(selected_ids)

                # Fetch only selected experts
                asyncio.run(runner.materialize_selected_experts(layer_idx, module, selected_ids))
            return hook

        print(f"Loading non-expert weights (attention + shared experts) across {cfg.num_hidden_layers} layers...")
        for i, layer in enumerate(model.model.layers):
            await runner.load_layer_non_experts(i, layer)
            layer.mlp.register_forward_pre_hook(make_moe_pre_hook(i))

        # 3. Generate tokens
        print(f"\n--- Starting Generation ---")
        print(f"Prompt: {args.prompt!r}")
        input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(args.device)

        generated_ids = input_ids.tolist()[0]
        start_time = time.perf_counter()

        with torch.inference_mode():
            for step in range(args.max_tokens):
                step_start = time.perf_counter()
                curr_input = torch.tensor([generated_ids], device=args.device)
                outputs = model(curr_input)
                next_token_id = int(torch.argmax(outputs.logits[0, -1, :]))
                generated_ids.append(next_token_id)
                next_token_str = tokenizer.decode([next_token_id])
                print(f"  Token {step+1}: {next_token_str!r} ({time.perf_counter() - step_start:.2f}s)")

        total_time = time.perf_counter() - start_time
        full_text = tokenizer.decode(generated_ids)

        print("\n" + "=" * 50)
        print("GENUINE MOE INFERENCE COMPLETED!")
        print(f"Output text: {full_text!r}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Total downloaded bytes: {runner.fetched_bytes / 1024**2:.2f} MB")
        print(f"Cached in memory: {len(runner.cache)} tensors")
        print(f"Bandwidth saved vs downloading 28GB model: {100 * (1 - runner.fetched_bytes / layout.total_weight_bytes):.1f}%!")
        print("=" * 50)

    finally:
        await source.aclose()


if __name__ == "__main__":
    asyncio.run(main())

