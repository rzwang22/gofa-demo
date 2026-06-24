import json
import sys
from pathlib import Path

import torch

GOFA_ROOT = Path("/mnt/sevenT/wrz_data/GOFA")
sys.path.insert(0, str(GOFA_ROOT))

from modules.gofa import GOFAMistralConfig, TrainingArguments, ModelArguments
from modules.gofa import GOFAMistral
from modules.utils import prepare_gofa_graph_input, prepare_gofa_graph_input_from_pyg


graph_type = "pyg"
device = torch.device("cuda")

if graph_type == "json":
    graph_path = GOFA_ROOT / "sample_graph.json"
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    gofa_input_graph = prepare_gofa_graph_input(graph, device=device)
elif graph_type == "pyg":
    graph_path = GOFA_ROOT / "sample_graph_pyg.pth"
    graph = torch.load(graph_path)
    gofa_input_graph = prepare_gofa_graph_input_from_pyg(graph, device=device)
else:
    raise ValueError("Unknown graph type")

model_args, training_args, gofa_args = ModelArguments(), TrainingArguments(), GOFAMistralConfig()

model_args.model_name_or_path = "/mnt/sevenT/wrz_data/GOFA/cache_data/model/Mistral-7B-Instruct-v0.2"
model_args.checkpoint_dir = "/mnt/sevenT/wrz_data/GOFA/cache_data/model"
model_args.dec_lora = False
model_args.lora_r = 0

gofa = GOFAMistral((model_args, training_args, gofa_args))
gofa.load_pretrained("/mnt/sevenT/wrz_data/GOFA/cache_data/model/mistral_qamag03_best_ckpt.pth")

model = gofa.model if hasattr(gofa, "model") else gofa

zeroed_lora = []
with torch.no_grad():
    for name, param in model.named_parameters():
        if "lora_" in name.lower():
            param.zero_()
            zeroed_lora.append(name)

print("=" * 100)
print("[INFO] Live GOFA clean inference")
print(f"[INFO] graph_path: {graph_path}")
print("[INFO] base_model: /mnt/sevenT/wrz_data/GOFA/cache_data/model/Mistral-7B-Instruct-v0.2")
print("[INFO] graph_layer_ckpt: /mnt/sevenT/wrz_data/GOFA/cache_data/model/mistral_qamag03_best_ckpt.pth")
print(f"[INFO] zeroed_lora_params: {len(zeroed_lora)}")
print("=" * 100)

gofa.to(device)

output = gofa.generate(gofa_input_graph)

print("=" * 100)
print("[LIVE_GOFA_OUTPUT]")
print(output)
print("=" * 100)
