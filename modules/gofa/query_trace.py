import json
import math
import os
from datetime import datetime
from pathlib import Path


TRACE_FORMAT = "gofa_query_trace"
TRACE_VERSION = 1
FORMAL_MEMORY_BITS = 4
FORMAL_KEY_BITS = 2
FORMAL_VALUE_BITS = 2
FORMAL_BATCH_ERROR = (
    "GOFA formal query trace currently supports batch_size=1 only because symbolic "
    "NODEID assignment is batch dependent."
)


def _jsonify(value):
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _jsonify(value.detach().cpu().tolist())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        try:
            return _jsonify(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, set):
        return [_jsonify(item) for item in sorted(value)]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _int_list(value):
    raw = _jsonify(value)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = [raw]

    result = []

    def visit(item):
        if isinstance(item, list):
            for nested in item:
                visit(nested)
        elif isinstance(item, bool):
            return
        elif isinstance(item, (int, float)):
            result.append(int(item))

    visit(raw)
    return result


def _shape(value):
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("shape", "orig_shape", "tensor_shape"):
            candidate = value.get(key)
            if isinstance(candidate, (list, tuple)):
                return [int(dim) for dim in candidate]
        if "tensor" in value:
            return _shape(value["tensor"])
        return None
    candidate = getattr(value, "shape", None)
    if candidate is not None:
        return [int(dim) for dim in candidate]
    return None


def _shape_numel(shape):
    if shape is None:
        return 0
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return int(numel)


def quantized_data_bytes(shape, bits):
    return int(math.ceil(_shape_numel(shape) * int(bits) / 8.0))


def _component_has_tokens(layer_kv, component):
    if not isinstance(layer_kv, dict):
        return False
    shape = _shape(layer_kv.get(component))
    return shape is not None and _shape_numel(shape) > 0 and (len(shape) < 2 or int(shape[-2]) > 0)


def infer_graph_batch_size(graph):
    if graph is None:
        return 0
    num_graphs = getattr(graph, "num_graphs", None)
    if num_graphs is not None:
        try:
            return int(num_graphs)
        except (TypeError, ValueError):
            pass
    ptr = _int_list(getattr(graph, "ptr", None))
    if ptr:
        return max(len(ptr) - 1, 0)
    batch = _int_list(getattr(graph, "batch", None))
    if batch:
        return max(batch) + 1
    return 1


class GOFAQueryTraceExporter:
    def __init__(self, owner, config, enabled=True):
        self.owner = owner
        self.config = dict(config)
        self.enabled = bool(enabled)
        self.queries_written = 0
        self.context = {}
        self.next_query_id = 0
        if not self.enabled:
            return
        output_dir = self.config.get("output_dir", "")
        if not output_dir:
            raise ValueError("gofa_query_trace.enabled=True requires gofa_query_trace.output_dir.")
        os.makedirs(output_dir, exist_ok=True)
        self.next_query_id = self._discover_next_query_id()

    def _discover_next_query_id(self):
        output_dir = self.config["output_dir"]
        maximum = -1
        for path in Path(output_dir).glob("query_*.json"):
            try:
                maximum = max(maximum, int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
        index_path = os.path.join(output_dir, "trace_index.jsonl")
        if os.path.isfile(index_path):
            with open(index_path) as handle:
                for line in handle:
                    try:
                        query_id = json.loads(line).get("query_id")
                        if isinstance(query_id, str) and query_id.startswith("query_"):
                            maximum = max(maximum, int(query_id.rsplit("_", 1)[-1]))
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue
        return maximum + 1

    def active(self):
        maximum = int(self.config.get("max_queries", 0))
        return self.enabled and (maximum == 0 or self.queries_written < maximum)

    def set_context(self, task_name=None, dataset_name=None, split=None, runtime_query_index=None):
        self.context = {
            "task_name": task_name,
            "dataset_name": dataset_name if dataset_name is not None else task_name,
            "split": split,
            "runtime_query_index": runtime_query_index,
        }

    def validate_graph(self, graph):
        if not self.enabled:
            return False
        batch_size = infer_graph_batch_size(graph)
        if batch_size != 1:
            raise RuntimeError(FORMAL_BATCH_ERROR)
        return self.active()

    def validate_setup(self):
        if not self.enabled:
            return
        owner = self.owner
        errors = []
        if not bool(owner.encoder_cache_enabled) or owner.encoder_cache_mode != "memory_kv":
            errors.append("use_encoder_cache=True and encoder_cache_mode=memory_kv are required")
        if not bool(owner.scheme_b_quant_enabled):
            errors.append("scheme_b_quant.enabled=True is required")
        else:
            expected = {
                "memory_base_bits": FORMAL_MEMORY_BITS,
                "key_base_bits": FORMAL_KEY_BITS,
                "value_base_bits": FORMAL_VALUE_BITS,
            }
            for key, value in expected.items():
                if int(owner.scheme_b_quant.get(key, -1)) != value:
                    errors.append(f"scheme_b_quant.{key} must be {value}")
        if not bool(getattr(owner.model_args, "encoder_cache_skip_nog", False)):
            errors.append("encoder_cache_skip_nog=True is required so NOG remains online")
        base_model = owner.model.icae.get_base_model().model
        suffix_layer_ids = list(range(base_model.gnn_start_layer, base_model.config.num_hidden_layers))
        if suffix_layer_ids != list(range(26, 32)):
            errors.append(f"suffix Transformer layers must be 26-31, got {suffix_layer_ids}")
        if errors:
            raise ValueError("Invalid GOFA formal query trace configuration: " + "; ".join(errors) + ".")

    def _write_json_atomic(self, path, payload):
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)

    def _default_task_name(self):
        task_names = self.owner._encoder_cache_manifest_task_names()
        if isinstance(task_names, list) and len(task_names) == 1:
            return task_names[0]
        if isinstance(task_names, str):
            return task_names
        return None

    def _text_preview(self, token_ids, text_len):
        if not self.config.get("include_text_preview", True):
            return None
        try:
            text = self.owner.model.tokenizer.decode(token_ids[:text_len], skip_special_tokens=True)
            return str(text).replace("\n", " ")[:120]
        except Exception:
            return None

    def _layer_shapes(self, cache_item, base_payload, suffix_layer_ids):
        base_layers = base_payload.get("text_kv", []) if isinstance(base_payload, dict) else []
        item_layers = cache_item.get("text_kv", []) if isinstance(cache_item, dict) else []
        result = []
        for offset, layer_id in enumerate(suffix_layer_ids):
            layer = base_layers[offset] if offset < len(base_layers) else None
            if not isinstance(layer, dict):
                layer = item_layers[offset] if offset < len(item_layers) else None
            result.append({
                "layer_id": int(layer_id),
                "key_shape": _shape(layer.get("key")) if isinstance(layer, dict) else None,
                "value_shape": _shape(layer.get("value")) if isinstance(layer, dict) else None,
            })
        return result

    def _inventory(
        self,
        token_ids,
        cache_items,
        quant_base_payloads,
        cache_keys,
        skip_cache_indices,
        num_node_text_items,
        suffix_layer_ids,
    ):
        inventory = []
        skip_set = set(skip_cache_indices)
        for item_index, ids in enumerate(token_ids):
            cache_item = cache_items[item_index] if item_index < len(cache_items) else None
            base_payload = quant_base_payloads[item_index] if item_index < len(quant_base_payloads) else None
            is_nog = item_index in skip_set
            item_type = "NOG" if is_nog else ("node" if item_index < num_node_text_items else "edge")
            text_len = None
            for source in (base_payload, cache_item):
                if isinstance(source, dict) and source.get("text_len") is not None:
                    text_len = int(source["text_len"])
                    break
            if text_len is None:
                text_len = max(len(ids) - int(self.owner.mem_size), 0)
            memory_source = None
            if isinstance(base_payload, dict):
                memory_source = base_payload.get("memory_state")
            if memory_source is None and isinstance(cache_item, dict):
                memory_source = cache_item.get("memory_state")
            entry = {
                "item_index": int(item_index),
                "item_type": item_type,
                "cache_key": cache_keys[item_index],
                "cache_eligible": not is_nog,
                "is_nog": is_nog,
                "sequence_length": int(len(ids)),
                "text_length": int(text_len),
                "memory_bits": FORMAL_MEMORY_BITS,
                "key_bits": FORMAL_KEY_BITS,
                "value_bits": FORMAL_VALUE_BITS,
                "memory_shape": _shape(memory_source),
                "text_kv_shapes": self._layer_shapes(cache_item, base_payload, suffix_layer_ids),
            }
            if self.config.get("include_token_ids", False):
                entry["token_ids"] = [int(token_id) for token_id in ids]
            if self.config.get("include_text_preview", True):
                entry["text_preview"] = self._text_preview(ids, text_len)
            inventory.append(entry)
        return inventory

    def _selective_access(self, cache_items, eligible_item_indices, skip_cache_indices, policy_details, suffix_layer_ids):
        eligible = sorted(set(int(index) for index in eligible_item_indices) - set(skip_cache_indices))
        by_key = {str(layer_id): [] for layer_id in suffix_layer_ids}
        by_value = {str(layer_id): [] for layer_id in suffix_layer_ids}
        for item_index in eligible:
            item = cache_items[item_index]
            layers = item.get("text_kv", []) if isinstance(item, dict) else []
            for offset, layer_id in enumerate(suffix_layer_ids):
                if offset >= len(layers):
                    continue
                if _component_has_tokens(layers[offset], "key"):
                    by_key[str(layer_id)].append(item_index)
                if _component_has_tokens(layers[offset], "value"):
                    by_value[str(layer_id)].append(item_index)
        selected_key = sorted(set().union(*(set(values) for values in by_key.values()))) if by_key else []
        selected_value = sorted(set().union(*(set(values) for values in by_value.values()))) if by_value else []
        complete = sorted(set(selected_key) & set(selected_value))
        total_items = len(cache_items)
        selected_union = set(selected_key) | set(selected_value)
        return {
            "policy_name": policy_details.get("kv_base_load_policy", "all") if isinstance(policy_details, dict) else "all",
            "eligible_item_indices": eligible,
            "selected_key_item_indices": selected_key,
            "selected_value_item_indices": selected_value,
            "complete_kv_item_indices": complete,
            "k_only_item_indices": sorted(set(selected_key) - set(selected_value)),
            "v_only_item_indices": sorted(set(selected_value) - set(selected_key)),
            "skipped_item_indices": sorted(set(range(total_items)) - selected_union),
            "key_item_mask": [index in set(selected_key) for index in range(total_items)],
            "value_item_mask": [index in set(selected_value) for index in range(total_items)],
            "effective_key_items_by_layer": by_key,
            "effective_value_items_by_layer": by_value,
            "selection_is_shared_across_suffix_layers": (
                len({tuple(values) for values in by_key.values()}) <= 1
                and len({tuple(values) for values in by_value.values()}) <= 1
            ),
        }

    def _traffic(self, inventory, selective):
        inventory_by_index = {item["item_index"]: item for item in inventory}
        cacheable = [item for item in inventory if item["cache_eligible"]]
        memory_bytes = sum(quantized_data_bytes(item["memory_shape"], FORMAL_MEMORY_BITS) for item in cacheable)

        def layer_component_bytes(item, layer_id, component, bits):
            for layer in item["text_kv_shapes"]:
                if int(layer["layer_id"]) == int(layer_id):
                    return quantized_data_bytes(layer[f"{component}_shape"], bits)
            return 0

        full_key_bytes = 0
        full_value_bytes = 0
        edge_cache_bytes = 0
        for item in cacheable:
            item_bytes = quantized_data_bytes(item["memory_shape"], FORMAL_MEMORY_BITS)
            for layer in item["text_kv_shapes"]:
                key_bytes = quantized_data_bytes(layer["key_shape"], FORMAL_KEY_BITS)
                value_bytes = quantized_data_bytes(layer["value_shape"], FORMAL_VALUE_BITS)
                full_key_bytes += key_bytes
                full_value_bytes += value_bytes
                item_bytes += key_bytes + value_bytes
            if item["item_type"] == "edge":
                edge_cache_bytes += item_bytes

        selected_key_bytes = 0
        selected_value_bytes = 0
        for layer_id, item_indices in selective["effective_key_items_by_layer"].items():
            for item_index in item_indices:
                selected_key_bytes += layer_component_bytes(
                    inventory_by_index[item_index], layer_id, "key", FORMAL_KEY_BITS
                )
        for layer_id, item_indices in selective["effective_value_items_by_layer"].items():
            for item_index in item_indices:
                selected_value_bytes += layer_component_bytes(
                    inventory_by_index[item_index], layer_id, "value", FORMAL_VALUE_BITS
                )
        persistent = memory_bytes + full_key_bytes + full_value_bytes
        loaded = memory_bytes + selected_key_bytes + selected_value_bytes
        return {
            "byte_accounting": "quantized_data_only_excluding_scale_and_container_metadata",
            "memory_cache_bytes": int(memory_bytes),
            "selected_key_bytes": int(selected_key_bytes),
            "selected_value_bytes": int(selected_value_bytes),
            "full_key_bytes": int(full_key_bytes),
            "full_value_bytes": int(full_value_bytes),
            "edge_cache_bytes": int(edge_cache_bytes),
            "nog_online_item_count": sum(1 for item in inventory if item["is_nog"]),
            "persistent_cache_bytes": int(persistent),
            "runtime_loaded_cache_bytes": int(loaded),
        }

    def _summary(self, inventory, selective, traffic, suffix_layer_ids):
        eligible_count = len(selective["eligible_item_indices"])
        layer_count = len(suffix_layer_ids)
        denominator = eligible_count * layer_count
        key_accesses = sum(len(values) for values in selective["effective_key_items_by_layer"].values())
        value_accesses = sum(len(values) for values in selective["effective_value_items_by_layer"].values())
        complete_accesses = 0
        for layer_id in suffix_layer_ids:
            complete_accesses += len(
                set(selective["effective_key_items_by_layer"][str(layer_id)])
                & set(selective["effective_value_items_by_layer"][str(layer_id)])
            )
        return {
            "total_item_count": len(inventory),
            "cacheable_item_count": sum(1 for item in inventory if item["cache_eligible"]),
            "NOG_count": sum(1 for item in inventory if item["is_nog"]),
            "selection_ratio_basis": "cacheable_item_layer_accesses",
            "selected_K_ratio": key_accesses / denominator if denominator else 0.0,
            "selected_V_ratio": value_accesses / denominator if denominator else 0.0,
            "selected_KV_ratio": complete_accesses / denominator if denominator else 0.0,
            "persistent_cache_bytes": traffic["persistent_cache_bytes"],
            "runtime_loaded_cache_bytes": traffic["runtime_loaded_cache_bytes"],
        }

    def _build_trace(
        self,
        graph,
        token_ids,
        cache_items,
        skip_cache_indices,
        eligible_item_indices,
        kv_base_policy_details,
        quant_base_payloads,
        quant_cache_keys,
        runtime_operation_shapes,
    ):
        owner = self.owner
        base_model = owner.model.icae.get_base_model().model
        suffix_layer_ids = list(range(base_model.gnn_start_layer, base_model.config.num_hidden_layers))
        num_items = len(token_ids)
        num_node_text_items = int(getattr(graph, "num_node_feat", num_items))
        cache_keys = [
            quant_cache_keys[index] if index < len(quant_cache_keys) and quant_cache_keys[index] else owner._encoder_cache_key(ids)
            for index, ids in enumerate(token_ids)
        ]
        inventory = self._inventory(
            token_ids,
            cache_items,
            quant_base_payloads,
            cache_keys,
            skip_cache_indices,
            num_node_text_items,
            suffix_layer_ids,
        )
        selective = self._selective_access(
            cache_items,
            eligible_item_indices,
            skip_cache_indices,
            kv_base_policy_details,
            suffix_layer_ids,
        )
        traffic = self._traffic(inventory, selective)
        summary = self._summary(inventory, selective, traffic, suffix_layer_ids)
        node_map = _int_list(getattr(graph, "node_map", None))
        question_index = _int_list(getattr(graph, "question_index", None))
        num_heads = int(getattr(base_model.config, "num_attention_heads", 0))
        hidden_size = int(getattr(base_model.config, "hidden_size", 0))
        num_kv_heads = int(getattr(base_model.config, "num_key_value_heads", num_heads))
        head_dim = int(getattr(base_model.layers[base_model.gnn_start_layer].self_attn, "head_dim", 0))
        if not head_dim and num_heads:
            head_dim = hidden_size // num_heads
        task_name = self.context.get("task_name") or self._default_task_name()
        dataset_name = self.context.get("dataset_name") or task_name
        runtime_query_index = self.context.get("runtime_query_index")
        if runtime_query_index is None:
            runtime_query_index = self.queries_written
        edge_index = _jsonify(getattr(graph, "edge_index", None))
        num_structural_edges = 0
        edge_shape = _shape(getattr(graph, "edge_index", None))
        if edge_shape and len(edge_shape) == 2:
            num_structural_edges = int(edge_shape[1])
        return {
            "trace_format": TRACE_FORMAT,
            "trace_version": TRACE_VERSION,
            "query_id": f"query_{self.next_query_id:06d}",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "repository_commit_sha": owner._trace_source_audit_repo_commit(),
            "task_name": task_name,
            "dataset_name": dataset_name,
            "split": self.context.get("split"),
            "runtime_query_index": int(runtime_query_index),
            "batch_size": infer_graph_batch_size(graph),
            "cache_mode": owner.encoder_cache_mode,
            "cache_tag": owner.encoder_cache_namespace,
            "model_configuration": {
                "hidden_size": hidden_size,
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "head_dim": head_dim,
                "mem_size": int(owner.mem_size),
                "gnn_start_layer": int(base_model.gnn_start_layer),
                "suffix_layer_ids": suffix_layer_ids,
            },
            "query_graph_structure": {
                "num_graph_nodes": len(node_map),
                "num_node_text_items": num_node_text_items,
                "num_edge_text_items": max(num_items - num_node_text_items, 0),
                "num_structural_edges": num_structural_edges,
                "target_index": _jsonify(getattr(graph, "target_index", None)),
                "question_index": question_index,
                "nog_local_index": question_index[0] if len(question_index) == 1 else None,
                "nog_local_indices": question_index,
                "node_map": node_map,
                "node_map_semantics": "graph_local_node_to_encoder_text_item",
                "edge_index": edge_index,
                "edge_map": _int_list(getattr(graph, "edge_map", None)),
                "edge_map_semantics": "structural_edge_to_edge_text_item",
                "batch": _int_list(getattr(graph, "batch", None)),
                "ptr": _int_list(getattr(graph, "ptr", None)),
            },
            "cache_item_inventory": inventory,
            "selective_kv_access": selective,
            "runtime_operation_shapes": _jsonify(runtime_operation_shapes),
            "traffic_metadata": traffic,
            "summary": summary,
        }

    def write_query(self, **kwargs):
        if not self.active():
            return None
        try:
            self.validate_graph(kwargs.get("graph"))
            trace = self._build_trace(**kwargs)
            query_id = trace["query_id"]
            filename = f"{query_id}.json"
            output_path = os.path.join(self.config["output_dir"], filename)
            self._write_json_atomic(output_path, trace)
            graph = trace["query_graph_structure"]
            index_entry = {
                "query_id": query_id,
                "task": trace["task_name"],
                "split": trace["split"],
                "trace_path": filename,
                "num_nodes": graph["num_graph_nodes"],
                "num_edges": graph["num_structural_edges"],
                "cacheable_items": trace["summary"]["cacheable_item_count"],
                "selected_K_items": len(trace["selective_kv_access"]["selected_key_item_indices"]),
                "selected_V_items": len(trace["selective_kv_access"]["selected_value_item_indices"]),
                "runtime_loaded_bytes": trace["summary"]["runtime_loaded_cache_bytes"],
            }
            with open(os.path.join(self.config["output_dir"], "trace_index.jsonl"), "a") as handle:
                handle.write(json.dumps(index_entry, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
            self.queries_written += 1
            self.next_query_id += 1
            print(
                "GOFA query trace written: "
                f"query_id={query_id}, task={trace['task_name']}, split={trace['split']}, "
                f"items={trace['summary']['total_item_count']}, "
                f"selected_K={index_entry['selected_K_items']}, selected_V={index_entry['selected_V_items']}, "
                f"path={output_path}"
            )
            return output_path
        except Exception as exc:
            if self.config.get("strict", True):
                raise RuntimeError(f"GOFA query trace export failed: {exc}") from exc
            print(f"GOFA query trace export warning: {exc}")
            return None
