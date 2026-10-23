"""List candidate small real checkpoints and their safetensors files.

Read-only reconnaissance: this only queries the public model API and prints
metadata.  It never downloads a weight byte.
"""

import httpx

CANDS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "EleutherAI/pythia-70m",
    "HuggingFaceTB/SmolLM-135M",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen3-0.6B",
    "unsloth/Llama-3.2-1B-Instruct",
]


def main() -> int:
    for repo in CANDS:
        try:
            resp = httpx.get(f"https://huggingface.co/api/models/{repo}", timeout=20)
            if resp.status_code != 200:
                print(repo, "HTTP", resp.status_code)
                continue
            d = resp.json()
            sha = (d.get("sha") or "")[:12]
            card = d.get("cardData") or {}
            lic = card.get("license")
            gated = d.get("gated")
            arch = (d.get("config") or {}).get("architectures")
            print(f"{repo}: sha={sha} license={lic} gated={gated} arch={arch}")
            for s in d.get("siblings", []):
                name = s["rfilename"]
                if name.endswith(".safetensors") or name in ("config.json",):
                    print("    ", name, s.get("size"))
        except Exception as exc:  # noqa: BLE001
            print(repo, "ERR", type(exc).__name__, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
