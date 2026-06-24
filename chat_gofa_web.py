import argparse
import json
import sys
from pathlib import Path

import torch

GOFA_ROOT = Path("/mnt/sevenT/wrz_data/GOFA")
sys.path.insert(0, str(GOFA_ROOT))

from modules.gofa import GOFAMistralConfig, TrainingArguments, ModelArguments
from modules.gofa import GOFAMistral
from modules.utils import prepare_gofa_graph_input, prepare_gofa_graph_input_from_pyg


def to_int(value):
    if torch.is_tensor(value):
        return int(value.flatten()[0].item())
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def get_index_value(value, idx):
    if torch.is_tensor(value):
        return int(value.flatten()[idx].item())
    if isinstance(value, (list, tuple)):
        return int(value[idx])
    return int(value)


def set_sequence_item(obj, attr_name, index, text):
    """
    Set obj.<attr_name>[index] = text for list/tuple-like text fields.
    """
    if not hasattr(obj, attr_name):
        return False, f"{attr_name}:missing"

    value = getattr(obj, attr_name)

    try:
        if isinstance(value, list):
            if len(value) <= index:
                return False, f"{attr_name}:index_out_of_range"
            value[index] = text
            return True, f"{attr_name}[{index}]"

        if isinstance(value, tuple):
            tmp = list(value)
            if len(tmp) <= index:
                return False, f"{attr_name}:index_out_of_range"
            tmp[index] = text
            setattr(obj, attr_name, tuple(tmp))
            return True, f"{attr_name}[{index}]"

        if isinstance(value, str):
            setattr(obj, attr_name, text)
            return True, f"{attr_name}:str"

        # Some custom containers support item assignment.
        value[index] = text
        return True, f"{attr_name}[{index}]:generic"

    except Exception as e:
        return False, f"{attr_name}:failed:{repr(e)}"


def patch_one_prepared_graph(g, question):
    """
    Patch the prepared GOFA graph object, not the original PyG graph.

    From your debug:
      question_index: [2]
      node_map: [1, 2, 0]
      graph node 2 -> feature 0
      g.x[0]: old question

    Therefore we update:
      - question[0] / questions[0]
      - answer[0] / answers[0], only because this sample graph stores the old prompt there too
      - x[feature_idx], where feature_idx = node_map[question_index[0]]
    """
    updated = []

    # 1. Update question-like fields.
    for attr in [
        "question",
        "questions",
        "question_text",
        "question_texts",
        "prompt",
        "prompts",
        "instruction",
        "instructions",
    ]:
        ok, msg = set_sequence_item(g, attr, 0, question)
        if ok:
            updated.append(msg)

    # 2. Update answer-like fields if they exist.
    # In your sample debug, answer[0] is identical to the old question.
    # This is metadata/debug input, not the generated model output.
    for attr in [
        "answer",
        "answers",
        "answer_text",
        "answer_texts",
    ]:
        ok, msg = set_sequence_item(g, attr, 0, question)
        if ok:
            updated.append(msg)

    # 3. Update node text corresponding to the question node.
    feature_idx = 0
    try:
        if hasattr(g, "question_index") and hasattr(g, "node_map"):
            q_node = to_int(getattr(g, "question_index"))
            node_map = getattr(g, "node_map")
            feature_idx = get_index_value(node_map, q_node)
    except Exception as e:
        updated.append(f"feature_idx_fallback_0:{repr(e)}")
        feature_idx = 0

    ok, msg = set_sequence_item(g, "x", feature_idx, question)
    if ok:
        updated.append(msg)
    else:
        updated.append(msg)

    return updated


def patch_prepared_graph(gofa_input_graph, question):
    """
    Handle single graph, list of graphs, or tuple of graphs.
    """
    all_updates = []

    if isinstance(gofa_input_graph, list):
        for i, g in enumerate(gofa_input_graph):
            updates = patch_one_prepared_graph(g, question)
            all_updates.append((f"graph[{i}]", updates))
        return all_updates

    if isinstance(gofa_input_graph, tuple):
        for i, g in enumerate(gofa_input_graph):
            updates = patch_one_prepared_graph(g, question)
            all_updates.append((f"graph[{i}]", updates))
        return all_updates

    updates = patch_one_prepared_graph(gofa_input_graph, question)
    all_updates.append(("graph", updates))
    return all_updates


def preview_prepared_graph(g):
    """
    Print a short preview before/after patching.
    """
    try:
        print("[PREVIEW] len(g.x):", len(g.x) if hasattr(g, "x") else "missing")
    except Exception as e:
        print("[PREVIEW] len(g.x) failed:", repr(e))

    for attr in ["question", "questions", "answer", "answers", "node_map", "question_index"]:
        try:
            if hasattr(g, attr):
                value = getattr(g, attr)
                print(f"[PREVIEW] {attr}: {value}")
        except Exception as e:
            print(f"[PREVIEW] {attr} failed: {repr(e)}")

    try:
        if hasattr(g, "x"):
            print("[PREVIEW] first 3 x:")
            for i in range(min(3, len(g.x))):
                print(f"  x[{i}]: {g.x[i]}")
    except Exception as e:
        print("[PREVIEW] x preview failed:", repr(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--graph-type", type=str, default="pyg", choices=["pyg", "json"])
    parser.add_argument("--graph-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.graph_type == "json":
        graph_path = Path(args.graph_path) if args.graph_path else GOFA_ROOT / "sample_graph.json"
        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)

        gofa_input_graph = prepare_gofa_graph_input(graph, device=device)

    elif args.graph_type == "pyg":
        graph_path = Path(args.graph_path) if args.graph_path else GOFA_ROOT / "sample_graph_pyg.pth"
        graph = torch.load(graph_path)

        gofa_input_graph = prepare_gofa_graph_input_from_pyg(graph, device=device)

    else:
        raise ValueError("Unknown graph type")

    print("=" * 100)
    print("[WEB_INPUT]")
    print(f"graph_path: {graph_path}")
    print(f"question: {args.question}")
    print("=" * 100)

    print("=" * 100)
    print("[BEFORE_PATCH_PREPARED_GRAPH]")
    preview_prepared_graph(gofa_input_graph)
    print("=" * 100)

    patch_updates = []
    if args.question:
        patch_updates = patch_prepared_graph(gofa_input_graph, args.question)

    print("=" * 100)
    print("[WEB_PATCH_AFTER_PREPARE]")
    print(f"patch_updates: {patch_updates}")
    print("=" * 100)

    print("=" * 100)
    print("[AFTER_PATCH_PREPARED_GRAPH]")
    preview_prepared_graph(gofa_input_graph)
    print("=" * 100)

    model_args, training_args, gofa_args = ModelArguments(), TrainingArguments(), GOFAMistralConfig()

    model_args.model_name_or_path = "/mnt/sevenT/wrz_data/GOFA/cache_data/model/Mistral-7B-Instruct-v0.2"
    model_args.checkpoint_dir = "/mnt/sevenT/wrz_data/GOFA/cache_data/model"
    model_args.dec_lora = False

    gofa = GOFAMistral((model_args, training_args, gofa_args))
    gofa.load_pretrained("/mnt/sevenT/wrz_data/GOFA/cache_data/model/mistral_qamag03_best_ckpt.pth")

    gofa.to(device)

    output = gofa.generate(gofa_input_graph)

    print("=" * 100)
    print("[LIVE_GOFA_OUTPUT]")
    print(output)
    print("=" * 100)


if __name__ == "__main__":
    main()
