import json, struct
from huggingface_hub import hf_hub_download
path = hf_hub_download("tiny-random/qwen3.5-moe", "model.safetensors")
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n)[:n].decode("utf-8"))
experts = [k for k in header if k.endswith(".weight") and "experts" in k]
print("expert tensor count", len(experts))
print(experts[:6])
proj_set = set()
for k in experts:
    tail = k.rsplit("experts.",1)[1]
    proj_set.add(tail.split(".")[1] + ".weight")
print("proj set", sorted(proj_set))
for k in experts[:3]:
    print(k, header[k]["shape"])

