#!/usr/bin/env python3
"""Validate formal GOFA one-query-per-trace JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modules.gofa.workload_profile import graph_signature, normalize_workload_profile, query_uid


MEMORY_BITS = 4
KEY_BITS = 2
VALUE_BITS = 2
SHAPE_FIELDS = (
    "q_projection_input_shape",
    "q_projection_output_shape",
    "qk_shape",
    "softmax_probability_shape",
    "pv_shape",
    "attention_output_shape",
    "mlp_input_shape",
    "mlp_output_shape",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def _valid_shape(shape: Any, *, allow_none: bool = False) -> bool:
    if shape is None:
        return allow_none
    return (
        isinstance(shape, list)
        and len(shape) > 0
        and all(isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape)
    )


def _numel(shape: list[int] | None) -> int:
    if shape is None:
        return 0
    result = 1
    for dim in shape:
        result *= dim
    return result


def _bytes(shape: list[int] | None, bits: int) -> int:
    return math.ceil(_numel(shape) * bits / 8)


def _scale_bytes(shape: list[int] | None) -> int:
    if not shape or _numel(shape) == 0:
        return 0
    return int(shape[-1]) * 4


def _layer_shape(item: dict[str, Any], layer_id: int, component: str) -> list[int] | None:
    for layer in item.get("text_kv_shapes", []):
        if isinstance(layer, dict) and layer.get("layer_id") == layer_id:
            return layer.get(f"{component}_shape")
    return None


def _check_no_cache_payload(value: Any, errors: list[str], path: str = "cache_item_inventory") -> None:
    forbidden = {"q", "q_packed", "tensor", "scale", "zero_point"}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in forbidden:
                errors.append(f"{path}.{key} contains a forbidden cache tensor payload")
            _check_no_cache_payload(item, errors, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_no_cache_payload(item, errors, f"{path}[{index}]")


def validate_trace(trace: dict[str, Any], source: str = "<memory>") -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(trace.get("trace_format") == "gofa_query_trace", "trace_format must be gofa_query_trace")
    require(trace.get("trace_version") == 1, "trace_version must be 1")
    require(trace.get("batch_size") == 1, "batch_size must be 1")
    for field in (
        "repository_commit_sha",
        "task_name",
        "dataset_name",
        "split",
        "runtime_query_index",
        "cache_mode",
        "cache_tag",
        "query_uid",
        "workload_profile",
        "sampling_hops",
        "sampling_max_nodes_per_hop",
        "graph_signature",
        "cache_item_count",
        "num_graph_nodes",
        "num_structural_edges",
        "logical_address_layout",
    ):
        require(field in trace, f"missing metadata field {field}")

    model = trace.get("model_configuration")
    require(isinstance(model, dict), "model_configuration must be an object")
    if not isinstance(model, dict):
        model = {}
    suffix_layers = model.get("suffix_layer_ids", [])
    require(suffix_layers == list(range(26, 32)), "suffix_layer_ids must be exactly 26-31")
    if isinstance(suffix_layers, list) and suffix_layers:
        require(
            suffix_layers == list(range(model.get("gnn_start_layer", -1), model.get("gnn_start_layer", -1) + 6)),
            "suffix_layer_ids must be contiguous from gnn_start_layer",
        )
    for field in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "mem_size"):
        require(isinstance(model.get(field), int) and model.get(field, 0) > 0, f"invalid model_configuration.{field}")

    graph = trace.get("query_graph_structure")
    require(isinstance(graph, dict), "query_graph_structure must be an object")
    if not isinstance(graph, dict):
        graph = {}
    inventory = trace.get("cache_item_inventory")
    require(isinstance(inventory, list), "cache_item_inventory must be an array")
    if not isinstance(inventory, list):
        inventory = []
    total_items = len(inventory)
    require(trace.get("cache_item_count") == total_items, "cache_item_count must equal inventory length")
    num_node_items = graph.get("num_node_text_items", 0)
    num_edge_items = graph.get("num_edge_text_items", 0)
    require(
        isinstance(num_node_items, int) and isinstance(num_edge_items, int)
        and num_node_items + num_edge_items == total_items,
        "node/edge text item counts must sum to cache inventory length",
    )
    node_map = graph.get("node_map", [])
    require(isinstance(node_map, list), "node_map must be an array")
    if isinstance(node_map, list):
        require(len(node_map) == graph.get("num_graph_nodes"), "node_map length must equal num_graph_nodes")
        require(
            all(isinstance(index, int) and 0 <= index < num_node_items for index in node_map),
            "node_map values must index graph-local node text items",
        )
    edge_map = graph.get("edge_map", [])
    require(isinstance(edge_map, list), "edge_map must be an array")
    if isinstance(edge_map, list):
        require(len(edge_map) == graph.get("num_structural_edges"), "edge_map length must equal num_structural_edges")
        require(
            all(isinstance(index, int) and 0 <= index < num_edge_items for index in edge_map),
            "edge_map values must index edge text items",
        )
    edge_index = graph.get("edge_index")
    require(
        isinstance(edge_index, list) and len(edge_index) == 2
        and all(isinstance(row, list) and len(row) == graph.get("num_structural_edges") for row in edge_index),
        "edge_index must have shape [2, num_structural_edges]",
    )
    require(graph.get("node_map_semantics") == "graph_local_node_to_encoder_text_item", "node_map semantics missing")
    require(graph.get("edge_map_semantics") == "structural_edge_to_edge_text_item", "edge_map semantics missing")
    require(trace.get("num_graph_nodes") == graph.get("num_graph_nodes"), "top-level num_graph_nodes mismatch")
    require(
        trace.get("num_structural_edges") == graph.get("num_structural_edges"),
        "top-level num_structural_edges mismatch",
    )
    question_indices = graph.get("question_index", [])
    require(isinstance(question_indices, list) and len(question_indices) == 1, "formal trace must have one question/NOG local index")
    target_indices = graph.get("target_index", [])
    require(
        isinstance(target_indices, list)
        and all(isinstance(index, int) and 0 <= index < graph.get("num_graph_nodes", 0) for index in target_indices),
        "target_index must contain graph-local node indices",
    )
    if isinstance(question_indices, list):
        require(
            all(isinstance(index, int) and 0 <= index < graph.get("num_graph_nodes", 0) for index in question_indices),
            "question_index must contain graph-local node indices",
        )
    require(graph.get("nog_local_indices") == question_indices, "NOG local indices must match question_index")
    require(
        graph.get("nog_local_index") == (question_indices[0] if isinstance(question_indices, list) and len(question_indices) == 1 else None),
        "NOG local index mismatch",
    )
    batch = graph.get("batch", [])
    ptr = graph.get("ptr", [])
    require(batch == [0] * graph.get("num_graph_nodes", 0), "batch vector must describe one graph")
    require(ptr == [0, graph.get("num_graph_nodes", 0)], "ptr must describe one graph")
    try:
        profile = normalize_workload_profile(trace.get("workload_profile"))
        require(trace.get("sampling_hops") == profile["hops"], "sampling_hops must match workload_profile")
        require(
            trace.get("sampling_max_nodes_per_hop") == profile["max_nodes_per_hop"],
            "sampling_max_nodes_per_hop must match workload_profile",
        )
    except (TypeError, ValueError, KeyError) as exc:
        errors.append(f"invalid workload_profile: {exc}")
    expected_signature = graph_signature(
        trace.get("task_name"),
        trace.get("split"),
        trace.get("runtime_query_index", 0),
        node_map=graph.get("node_map"),
        edge_map=graph.get("edge_map"),
        edge_index=graph.get("edge_index"),
        target_index=graph.get("target_index"),
        question_index=graph.get("question_index"),
    )
    require(trace.get("graph_signature") == expected_signature, "graph_signature mismatch")
    require(
        trace.get("query_uid") == query_uid(
            trace.get("task_name"),
            trace.get("split"),
            trace.get("runtime_query_index", 0),
            expected_signature,
        ),
        "query_uid mismatch",
    )
    if isinstance(edge_index, list) and len(edge_index) == 2:
        require(
            all(
                isinstance(index, int) and 0 <= index < graph.get("num_graph_nodes", 0)
                for row in edge_index if isinstance(row, list)
                for index in row
            ),
            "edge_index must contain graph-local node indices",
        )

    layout = trace.get("logical_address_layout")
    if isinstance(layout, dict):
        require(
            layout.get("address_space") == "query_local_cache_and_gather_metadata_bytes",
            "logical layout address_space mismatch",
        )
        alignment = layout.get("alignment_bytes")
        components = layout.get("components")
        require(isinstance(alignment, int) and alignment > 0, "logical layout alignment must be positive")
        require(isinstance(components, list), "logical layout components must be an array")
        previous_end = 0
        for index, component in enumerate(components or []):
            if not isinstance(component, dict):
                errors.append(f"logical layout component {index} must be an object")
                continue
            base = component.get("base_offset")
            size = component.get("size_bytes")
            require(isinstance(base, int) and base >= previous_end, f"logical component {index} overlaps")
            require(isinstance(size, int) and size > 0, f"logical component {index} has invalid size")
            if isinstance(base, int) and isinstance(alignment, int) and alignment > 0:
                require(base % alignment == 0, f"logical component {index} is not aligned")
            if isinstance(base, int) and isinstance(size, int):
                previous_end = base + size
        require(
            isinstance(layout.get("total_size_bytes"), int)
            and layout.get("total_size_bytes", -1) >= previous_end,
            "logical layout total_size_bytes is too small",
        )
        expected_components = []
        for item in inventory:
            if not isinstance(item, dict) or not item.get("cache_eligible"):
                continue
            memory_size = _bytes(item.get("memory_shape"), int(item.get("memory_bits", MEMORY_BITS)))
            if memory_size:
                expected_components.append((item.get("item_index"), "memory", None, "data", memory_size, None, None))
            memory_scale_size = _scale_bytes(item.get("memory_shape"))
            if memory_scale_size:
                expected_components.append(
                    (item.get("item_index"), "memory", None, "scale", memory_scale_size, None, None)
                )
            for layer in item.get("text_kv_shapes", []):
                if not isinstance(layer, dict):
                    continue
                for component, bits in (("key", KEY_BITS), ("value", VALUE_BITS)):
                    size = _bytes(layer.get(f"{component}_shape"), int(item.get(f"{component}_bits", bits)))
                    if size:
                        expected_components.append(
                            (item.get("item_index"), component, layer.get("layer_id"), "data", size, None, None)
                        )
                    scale_size = _scale_bytes(layer.get(f"{component}_shape"))
                    if scale_size:
                        expected_components.append(
                            (item.get("item_index"), component, layer.get("layer_id"), "scale", scale_size, None, None)
                        )
        layout_access = trace.get("selective_kv_access", {})
        layout_eligible = layout_access.get("eligible_item_indices", []) if isinstance(layout_access, dict) else []
        layout_by_key = layout_access.get("effective_key_items_by_layer", {}) if isinstance(layout_access, dict) else {}
        layout_by_value = layout_access.get("effective_value_items_by_layer", {}) if isinstance(layout_access, dict) else {}
        if layout_eligible:
            expected_components.append(
                (None, "memory_item_indices", None, "gather_index_metadata", 4 * len(layout_eligible), "uint32", layout_eligible)
            )
        for layer_id in suffix_layers:
            indices = layout_by_key.get(str(layer_id), []) if isinstance(layout_by_key, dict) else []
            if indices:
                expected_components.append(
                    (None, "selected_key_item_indices", layer_id, "gather_index_metadata", 4 * len(indices), "uint32", indices)
                )
        for layer_id in suffix_layers:
            indices = layout_by_value.get(str(layer_id), []) if isinstance(layout_by_value, dict) else []
            if indices:
                expected_components.append(
                    (None, "selected_value_item_indices", layer_id, "gather_index_metadata", 4 * len(indices), "uint32", indices)
                )
        actual_components = [
            (
                component.get("item_index"),
                component.get("component"),
                component.get("layer_id"),
                component.get("storage_kind"),
                component.get("size_bytes"),
                component.get("index_dtype"),
                component.get("item_indices"),
            )
            for component in (components or [])
            if isinstance(component, dict)
        ]
        require(
            actual_components == expected_components,
            "logical layout must contain every cache data/scale and gather/index component in order",
        )

    inventory_by_index: dict[int, dict[str, Any]] = {}
    nog_indices: list[int] = []
    for expected_index, item in enumerate(inventory):
        require(isinstance(item, dict), f"inventory[{expected_index}] must be an object")
        if not isinstance(item, dict):
            continue
        require(item.get("item_index") == expected_index, f"inventory[{expected_index}] item_index mismatch")
        require(isinstance(item.get("cache_key"), str) and bool(item.get("cache_key")), f"inventory[{expected_index}] cache_key missing")
        inventory_by_index[expected_index] = item
        is_nog = item.get("is_nog") is True
        if is_nog:
            nog_indices.append(expected_index)
            require(item.get("item_type") == "NOG", f"inventory[{expected_index}] NOG item_type mismatch")
            require(item.get("cache_eligible") is False, f"inventory[{expected_index}] NOG must not be cache eligible")
        else:
            require(item.get("cache_eligible") is True, f"inventory[{expected_index}] cached item must be eligible")
            expected_type = "node" if expected_index < num_node_items else "edge"
            require(item.get("item_type") == expected_type, f"inventory[{expected_index}] item_type must be {expected_type}")
        require(item.get("memory_bits") == MEMORY_BITS, f"inventory[{expected_index}] memory_bits must be 4")
        require(item.get("key_bits") == KEY_BITS, f"inventory[{expected_index}] key_bits must be 2")
        require(item.get("value_bits") == VALUE_BITS, f"inventory[{expected_index}] value_bits must be 2")
        require(_valid_shape(item.get("memory_shape")), f"inventory[{expected_index}] invalid memory_shape")
        require(
            item.get("memory_scale_shape") == [model.get("hidden_size")],
            f"inventory[{expected_index}] memory_scale_shape mismatch",
        )
        require(
            item.get("memory_shape") == [model.get("mem_size"), model.get("hidden_size")],
            f"inventory[{expected_index}] memory_shape does not match model dimensions",
        )
        layers = item.get("text_kv_shapes")
        require(isinstance(layers, list) and len(layers) == len(suffix_layers), f"inventory[{expected_index}] text_kv layer count mismatch")
        if isinstance(layers, list):
            require(
                all(isinstance(layer, dict) for layer in layers),
                f"inventory[{expected_index}] text_kv layers must be objects",
            )
            require(
                [layer.get("layer_id") for layer in layers if isinstance(layer, dict)] == suffix_layers,
                f"inventory[{expected_index}] layer ids mismatch",
            )
            for layer in layers:
                if not isinstance(layer, dict):
                    continue
                require(_valid_shape(layer.get("key_shape")), f"inventory[{expected_index}] invalid key shape")
                require(_valid_shape(layer.get("value_shape")), f"inventory[{expected_index}] invalid value shape")
                expected_kv_shape = [model.get("num_key_value_heads"), item.get("text_length"), model.get("head_dim")]
                require(layer.get("key_shape") == expected_kv_shape, f"inventory[{expected_index}] key shape mismatch")
                require(layer.get("value_shape") == expected_kv_shape, f"inventory[{expected_index}] value shape mismatch")
                require(
                    layer.get("key_scale_shape") == [model.get("head_dim")],
                    f"inventory[{expected_index}] key scale shape mismatch",
                )
                require(
                    layer.get("value_scale_shape") == [model.get("head_dim")],
                    f"inventory[{expected_index}] value scale shape mismatch",
                )
    _check_no_cache_payload(inventory, errors)
    if isinstance(question_indices, list) and len(question_indices) == 1 and isinstance(node_map, list):
        question_index = question_indices[0]
        if isinstance(question_index, int) and 0 <= question_index < len(node_map):
            require(node_map[question_index] in nog_indices, "question_index must map to the non-cacheable NOG item")
    require(len(nog_indices) == 1, "formal one-query trace must contain exactly one NOG item")

    access = trace.get("selective_kv_access")
    require(isinstance(access, dict), "selective_kv_access must be an object")
    if not isinstance(access, dict):
        access = {}
    eligible = access.get("eligible_item_indices", [])
    selected_key = access.get("selected_key_item_indices", [])
    selected_value = access.get("selected_value_item_indices", [])
    for field, values in (("eligible", eligible), ("selected_key", selected_key), ("selected_value", selected_value)):
        require(isinstance(values, list), f"{field} indices must be an array")
        if isinstance(values, list):
            require(len(values) == len(set(values)), f"{field} indices must be unique")
            require(all(isinstance(index, int) and 0 <= index < total_items for index in values), f"{field} index out of range")
    require(set(eligible) == {index for index, item in inventory_by_index.items() if item.get("cache_eligible")}, "eligible items mismatch inventory")
    require(set(selected_key) <= set(eligible), "selected key items must be eligible")
    require(set(selected_value) <= set(eligible), "selected value items must be eligible")
    require(len(access.get("key_item_mask", [])) == total_items, "key_item_mask length mismatch")
    require(len(access.get("value_item_mask", [])) == total_items, "value_item_mask length mismatch")
    require(
        [index for index, selected in enumerate(access.get("key_item_mask", [])) if selected] == selected_key,
        "key_item_mask does not match selected_key_item_indices",
    )
    require(
        [index for index, selected in enumerate(access.get("value_item_mask", [])) if selected] == selected_value,
        "value_item_mask does not match selected_value_item_indices",
    )
    by_key = access.get("effective_key_items_by_layer", {})
    by_value = access.get("effective_value_items_by_layer", {})
    expected_layer_keys = {str(layer_id) for layer_id in suffix_layers}
    require(isinstance(by_key, dict) and set(by_key) == expected_layer_keys, "effective key layer access mismatch")
    require(isinstance(by_value, dict) and set(by_value) == expected_layer_keys, "effective value layer access mismatch")
    valid_key_layers = isinstance(by_key, dict) and all(isinstance(values, list) for values in by_key.values())
    valid_value_layers = isinstance(by_value, dict) and all(isinstance(values, list) for values in by_value.values())
    require(valid_key_layers, "effective key layer values must be arrays")
    require(valid_value_layers, "effective value layer values must be arrays")
    if valid_key_layers:
        require(set().union(*(set(values) for values in by_key.values())) == set(selected_key), "selected key union mismatch")
    if valid_value_layers:
        require(set().union(*(set(values) for values in by_value.values())) == set(selected_value), "selected value union mismatch")
    shared = (
        valid_key_layers and valid_value_layers
        and len({tuple(values) for values in by_key.values()}) <= 1
        and len({tuple(values) for values in by_value.values()}) <= 1
    )
    require(access.get("selection_is_shared_across_suffix_layers") is shared, "shared layer selection flag mismatch")
    require(access.get("selection_is_shared_across_suffix_layers") is True, "formal K/V selection must be shared across suffix layers")
    require(set(access.get("complete_kv_item_indices", [])) == set(selected_key) & set(selected_value), "complete K/V items mismatch")
    require(set(access.get("k_only_item_indices", [])) == set(selected_key) - set(selected_value), "K-only items mismatch")
    require(set(access.get("v_only_item_indices", [])) == set(selected_value) - set(selected_key), "V-only items mismatch")

    runtime = trace.get("runtime_operation_shapes")
    require(isinstance(runtime, dict), "runtime_operation_shapes must be an object")
    if not isinstance(runtime, dict):
        runtime = {}
    item_order = runtime.get("item_order", [])
    require(isinstance(item_order, list) and len(item_order) == graph.get("num_graph_nodes", 0) + num_edge_items, "runtime item_order length mismatch")
    runtime_layers = runtime.get("layers", [])
    require(
        isinstance(runtime_layers, list)
        and all(isinstance(layer, dict) for layer in runtime_layers)
        and [layer.get("layer_id") for layer in runtime_layers] == suffix_layers,
        "runtime suffix layers mismatch",
    )
    if isinstance(runtime_layers, list):
        for layer in runtime_layers:
            if not isinstance(layer, dict):
                errors.append("runtime layer entry must be an object")
                continue
            layer_id = layer.get("layer_id")
            items = layer.get("items", [])
            require(isinstance(items, list) and len(items) == len(item_order), f"layer {layer_id} runtime item count mismatch")
            for runtime_index, item in enumerate(items if isinstance(items, list) else []):
                if not isinstance(item, dict):
                    errors.append(f"layer {layer_id} runtime item {runtime_index} must be an object")
                    continue
                require(item.get("runtime_item_index") == runtime_index, f"layer {layer_id} runtime item index mismatch")
                if runtime_index < len(item_order):
                    require(item.get("item_index") == item_order[runtime_index], f"layer {layer_id} source item mapping mismatch")
                for field in SHAPE_FIELDS:
                    require(_valid_shape(item.get(field)), f"layer {layer_id} item {runtime_index} invalid {field}")
                require(item.get("qk_shape") == item.get("softmax_probability_shape"), f"layer {layer_id} QK/softmax shape mismatch")
                require(
                    item.get("q_projection_input_shape") == [1, model.get("mem_size"), model.get("hidden_size")],
                    f"layer {layer_id} Q projection input dimensions mismatch",
                )
                require(
                    item.get("q_projection_output_shape") == [1, model.get("mem_size"), model.get("hidden_size")],
                    f"layer {layer_id} Q projection output dimensions mismatch",
                )
                require(
                    item.get("pv_shape") == [1, model.get("num_attention_heads"), model.get("mem_size"), model.get("head_dim")],
                    f"layer {layer_id} PV dimensions mismatch",
                )
                for field in ("attention_output_shape", "mlp_input_shape", "mlp_output_shape"):
                    require(
                        item.get(field) == [1, model.get("mem_size"), model.get("hidden_size")],
                        f"layer {layer_id} {field} dimensions mismatch",
                    )
            gnn = layer.get("gnn")
            require(isinstance(gnn, dict), f"layer {layer_id} gnn shapes missing")
            if isinstance(gnn, dict):
                for field in ("node_input_shape", "node_output_shape", "edge_input_shape"):
                    require(_valid_shape(gnn.get(field)), f"layer {layer_id} invalid gnn.{field}")
                require(
                    gnn.get("node_input_shape") == [graph.get("num_graph_nodes"), model.get("mem_size"), model.get("hidden_size")],
                    f"layer {layer_id} GNN node input dimensions mismatch",
                )
                require(gnn.get("node_output_shape") == gnn.get("node_input_shape"), f"layer {layer_id} GNN node output dimensions mismatch")
                require(
                    gnn.get("edge_input_shape") == [graph.get("num_structural_edges"), model.get("mem_size"), model.get("hidden_size")],
                    f"layer {layer_id} GNN edge input dimensions mismatch",
                )
                require(gnn.get("edge_output_shape") is None, f"layer {layer_id} edge output must be null")
                require(gnn.get("edge_output_status") == "not_produced_by_gofa_gnn_layer", f"layer {layer_id} edge output status mismatch")

    memory_bytes = sum(_bytes(item.get("memory_shape"), MEMORY_BITS) for item in inventory if item.get("cache_eligible"))
    full_key_bytes = 0
    full_value_bytes = 0
    memory_scale_bytes = sum(
        _scale_bytes(item.get("memory_shape")) for item in inventory if item.get("cache_eligible")
    )
    full_key_scale_bytes = 0
    full_value_scale_bytes = 0
    edge_cache_bytes = 0
    edge_cache_scale_bytes = 0
    for item in inventory:
        if not item.get("cache_eligible"):
            continue
        item_bytes = _bytes(item.get("memory_shape"), MEMORY_BITS)
        item_scale_bytes = _scale_bytes(item.get("memory_shape"))
        for layer in item.get("text_kv_shapes", []):
            if not isinstance(layer, dict):
                continue
            key_bytes = _bytes(layer.get("key_shape"), KEY_BITS)
            value_bytes = _bytes(layer.get("value_shape"), VALUE_BITS)
            full_key_bytes += key_bytes
            full_value_bytes += value_bytes
            key_scale_bytes = _scale_bytes(layer.get("key_shape"))
            value_scale_bytes = _scale_bytes(layer.get("value_shape"))
            full_key_scale_bytes += key_scale_bytes
            full_value_scale_bytes += value_scale_bytes
            item_bytes += key_bytes + value_bytes
            item_scale_bytes += key_scale_bytes + value_scale_bytes
        if item.get("item_type") == "edge":
            edge_cache_bytes += item_bytes
            edge_cache_scale_bytes += item_scale_bytes
    selected_key_bytes = sum(
        _bytes(_layer_shape(inventory_by_index[index], int(layer_id), "key"), KEY_BITS)
        for layer_id, indices in by_key.items()
        for index in indices
        if index in inventory_by_index
    ) if valid_key_layers else 0
    selected_value_bytes = sum(
        _bytes(_layer_shape(inventory_by_index[index], int(layer_id), "value"), VALUE_BITS)
        for layer_id, indices in by_value.items()
        for index in indices
        if index in inventory_by_index
    ) if valid_value_layers else 0
    selected_key_scale_bytes = sum(
        _scale_bytes(_layer_shape(inventory_by_index[index], int(layer_id), "key"))
        for layer_id, indices in by_key.items()
        for index in indices
        if index in inventory_by_index
    ) if valid_key_layers else 0
    selected_value_scale_bytes = sum(
        _scale_bytes(_layer_shape(inventory_by_index[index], int(layer_id), "value"))
        for layer_id, indices in by_value.items()
        for index in indices
        if index in inventory_by_index
    ) if valid_value_layers else 0
    key_access_count = sum(len(values) for values in by_key.values()) if valid_key_layers else 0
    value_access_count = sum(len(values) for values in by_value.values()) if valid_value_layers else 0
    persistent_data_bytes = memory_bytes + full_key_bytes + full_value_bytes
    runtime_data_bytes = memory_bytes + selected_key_bytes + selected_value_bytes
    persistent_scale_bytes = memory_scale_bytes + full_key_scale_bytes + full_value_scale_bytes
    runtime_scale_bytes = memory_scale_bytes + selected_key_scale_bytes + selected_value_scale_bytes
    bytes_per_index = 4
    memory_index_bytes = len(eligible) * bytes_per_index
    selected_key_index_bytes = key_access_count * bytes_per_index
    selected_value_index_bytes = value_access_count * bytes_per_index
    runtime_index_bytes = memory_index_bytes + selected_key_index_bytes + selected_value_index_bytes
    traffic = trace.get("traffic_metadata", {})
    expected_traffic = {
        "memory_cache_bytes": memory_bytes,
        "selected_key_bytes": selected_key_bytes,
        "selected_value_bytes": selected_value_bytes,
        "full_key_bytes": full_key_bytes,
        "full_value_bytes": full_value_bytes,
        "edge_cache_bytes": edge_cache_bytes,
        "nog_online_item_count": len(nog_indices),
        "persistent_cache_bytes": persistent_data_bytes,
        "runtime_loaded_cache_bytes": runtime_data_bytes,
        "runtime_loaded_scale_bytes": runtime_scale_bytes,
        "runtime_gather_index_metadata_bytes": runtime_index_bytes,
    }
    require(
        traffic.get("byte_accounting") == "separate_quantized_data_fp32_scale_and_uint32_gather_index",
        "traffic byte accounting mode mismatch",
    )
    for key, expected in expected_traffic.items():
        require(traffic.get(key) == expected, f"traffic_metadata.{key}={traffic.get(key)} expected {expected}")
    expected_data = {
        "memory_cache": memory_bytes,
        "selected_key": selected_key_bytes,
        "selected_value": selected_value_bytes,
        "full_key": full_key_bytes,
        "full_value": full_value_bytes,
        "edge_cache": edge_cache_bytes,
        "persistent_cache": persistent_data_bytes,
        "runtime_loaded": runtime_data_bytes,
    }
    expected_scales = {
        "dtype": "float32",
        "memory_cache": memory_scale_bytes,
        "selected_key": selected_key_scale_bytes,
        "selected_value": selected_value_scale_bytes,
        "full_key": full_key_scale_bytes,
        "full_value": full_value_scale_bytes,
        "edge_cache": edge_cache_scale_bytes,
        "persistent_cache": persistent_scale_bytes,
        "runtime_loaded": runtime_scale_bytes,
    }
    expected_indices = {
        "index_dtype": "uint32",
        "bytes_per_index": bytes_per_index,
        "memory_item_indices": memory_index_bytes,
        "selected_key_item_indices": selected_key_index_bytes,
        "selected_value_item_indices": selected_value_index_bytes,
        "runtime_loaded": runtime_index_bytes,
    }
    require(traffic.get("logical_data_bytes") == expected_data, "logical data byte category mismatch")
    require(traffic.get("scale_bytes") == expected_scales, "scale byte category mismatch")
    require(
        traffic.get("gather_index_metadata_bytes") == expected_indices,
        "gather/index metadata byte category mismatch",
    )

    summary = trace.get("summary", {})
    require(summary.get("total_item_count") == total_items, "summary total item count mismatch")
    require(summary.get("cacheable_item_count") == len(eligible), "summary cacheable item count mismatch")
    require(summary.get("NOG_count") == len(nog_indices), "summary NOG count mismatch")
    require(summary.get("selection_ratio_basis") == "cacheable_item_layer_accesses", "summary selection ratio basis mismatch")
    require(summary.get("persistent_cache_bytes") == expected_traffic["persistent_cache_bytes"], "summary persistent bytes mismatch")
    require(summary.get("runtime_loaded_cache_bytes") == expected_traffic["runtime_loaded_cache_bytes"], "summary loaded bytes mismatch")
    require(summary.get("persistent_scale_bytes") == persistent_scale_bytes, "summary persistent scale bytes mismatch")
    require(summary.get("runtime_loaded_scale_bytes") == runtime_scale_bytes, "summary runtime scale bytes mismatch")
    require(
        summary.get("runtime_gather_index_metadata_bytes") == runtime_index_bytes,
        "summary runtime gather/index bytes mismatch",
    )
    denominator = len(eligible) * len(suffix_layers)
    complete_access_count = sum(
        len(set(by_key.get(str(layer_id), [])) & set(by_value.get(str(layer_id), [])))
        for layer_id in suffix_layers
    ) if valid_key_layers and valid_value_layers else 0
    expected_ratios = {
        "selected_K_ratio": key_access_count / denominator if denominator else 0.0,
        "selected_V_ratio": value_access_count / denominator if denominator else 0.0,
        "selected_KV_ratio": complete_access_count / denominator if denominator else 0.0,
    }
    for key, expected in expected_ratios.items():
        actual = summary.get(key)
        require(isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), f"summary {key} mismatch")

    return [f"{source}: {error}" for error in errors]


def _trace_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("query_*.json"))
    if input_path.suffix == ".jsonl":
        paths = []
        with input_path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                trace_path = Path(entry["trace_path"])
                paths.append(trace_path if trace_path.is_absolute() else input_path.parent / trace_path)
        return paths
    return [input_path]


def _validate_index(directory: Path, traces: dict[str, dict[str, Any]]) -> list[str]:
    index_path = directory / "trace_index.jsonl"
    if not index_path.is_file():
        return [f"{index_path}: missing trace index"]
    errors = []
    entries = []
    with index_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{index_path}:{line_number}: invalid JSON: {exc}")
    current_entries = [entry for entry in entries if entry.get("query_id") in traces]
    if len(current_entries) != len(traces):
        errors.append(f"{index_path}: index has {len(current_entries)} matching entries for {len(traces)} trace files")
    for entry in current_entries:
        trace = traces[entry["query_id"]]
        graph = trace["query_graph_structure"]
        expected = {
            "task": trace["task_name"],
            "split": trace["split"],
            "num_nodes": graph["num_graph_nodes"],
            "num_edges": graph["num_structural_edges"],
            "cacheable_items": trace["summary"]["cacheable_item_count"],
            "selected_K_items": len(trace["selective_kv_access"]["selected_key_item_indices"]),
            "selected_V_items": len(trace["selective_kv_access"]["selected_value_item_indices"]),
            "runtime_loaded_bytes": trace["summary"]["runtime_loaded_cache_bytes"],
            "runtime_loaded_scale_bytes": trace["summary"]["runtime_loaded_scale_bytes"],
            "runtime_gather_index_metadata_bytes": trace["summary"]["runtime_gather_index_metadata_bytes"],
        }
        for key, value in expected.items():
            if entry.get(key) != value:
                errors.append(f"{index_path}: {entry['query_id']} field {key} mismatch")
        trace_path = Path(entry.get("trace_path", ""))
        resolved = trace_path if trace_path.is_absolute() else directory / trace_path
        if not resolved.is_file():
            errors.append(f"{index_path}: missing indexed trace {resolved}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Trace JSON, trace_index.jsonl, or trace directory")
    args = parser.parse_args()
    paths = _trace_paths(args.input)
    if not paths:
        raise RuntimeError(f"No query trace files found under {args.input}")
    all_errors = []
    traces = {}
    for path in paths:
        trace = _load_json(path)
        all_errors.extend(validate_trace(trace, str(path)))
        if isinstance(trace.get("query_id"), str):
            traces[trace["query_id"]] = trace
    if args.input.is_dir():
        all_errors.extend(_validate_index(args.input, traces))
    if all_errors:
        for error in all_errors:
            print(f"ERROR: {error}")
        print(f"GOFA query trace validation failed: traces={len(paths)}, errors={len(all_errors)}")
        return 1
    print(f"GOFA query trace validation passed: traces={len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
