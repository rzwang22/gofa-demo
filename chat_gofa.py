import torch

from modules.gofa import GOFAMistralConfig, TrainingArguments, ModelArguments

from modules.gofa import GOFAMistral
from modules.utils import prepare_gofa_graph_input, prepare_gofa_graph_input_from_pyg
import json

graph_type = "pyg"
device = torch.device("cuda")

if graph_type == "json":
    with open("sample_graph.json", "r") as f:
        graph = json.load(f)
    gofa_input_graph = prepare_gofa_graph_input(graph, device=device)
elif graph_type == "pyg":
    graph = torch.load("sample_graph_pyg.pth")
    gofa_input_graph = prepare_gofa_graph_input_from_pyg(graph, device=device)
else:
    raise ValueError("Unknown graph type")

model_args, training_args, gofa_args = ModelArguments(), TrainingArguments(), GOFAMistralConfig()

model_args.model_name_or_path = "/mnt/sevenT/wrz_data/GOFA/cache_data/model/Mistral-7B-Instruct-v0.2"
model_args.checkpoint_dir = "/mnt/sevenT/wrz_data/GOFA/cache_data/model"
model_args.dec_lora = False

gofa = GOFAMistral((model_args, training_args, gofa_args))
gofa.load_pretrained("/mnt/sevenT/wrz_data/GOFA/cache_data/model/mistral_qamag03_best_ckpt.pth")

gofa.to(device)
print(gofa.generate(gofa_input_graph))
