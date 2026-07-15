import copy
import unittest

from scripts.summarize_gofa_query_traces import summarize_traces
from scripts.validate_gofa_query_trace import validate_trace


def _synthetic_trace():
    suffix_layers = list(range(26, 32))
    inventory = []
    for item_index, item_type in enumerate(("node", "node", "NOG", "edge")):
        inventory.append({
            "item_index": item_index,
            "item_type": item_type,
            "cache_key": f"key-{item_index}",
            "cache_eligible": item_type != "NOG",
            "is_nog": item_type == "NOG",
            "sequence_length": 4,
            "text_length": 2,
            "memory_bits": 4,
            "key_bits": 2,
            "value_bits": 2,
            "memory_shape": [2, 4],
            "text_kv_shapes": [
                {"layer_id": layer_id, "key_shape": [1, 2, 4], "value_shape": [1, 2, 4]}
                for layer_id in suffix_layers
            ],
        })
    runtime_layers = []
    item_order = [0, 2, 3]
    for layer_id in suffix_layers:
        runtime_layers.append({
            "layer_id": layer_id,
            "items": [
                {
                    "runtime_item_index": runtime_index,
                    "item_index": item_index,
                    "q_projection_input_shape": [1, 2, 4],
                    "q_projection_output_shape": [1, 2, 4],
                    "qk_shape": [1, 1, 2, 4],
                    "softmax_probability_shape": [1, 1, 2, 4],
                    "pv_shape": [1, 1, 2, 4],
                    "attention_output_shape": [1, 2, 4],
                    "mlp_input_shape": [1, 2, 4],
                    "mlp_output_shape": [1, 2, 4],
                }
                for runtime_index, item_index in enumerate(item_order)
            ],
            "gnn": {
                "node_input_shape": [2, 2, 4],
                "node_output_shape": [2, 2, 4],
                "edge_input_shape": [2, 2, 4],
                "edge_output_shape": None,
                "edge_output_status": "not_produced_by_gofa_gnn_layer",
            },
        })
    by_key = {str(layer_id): [0, 3] for layer_id in suffix_layers}
    by_value = {str(layer_id): [0] for layer_id in suffix_layers}
    return {
        "trace_format": "gofa_query_trace",
        "trace_version": 1,
        "query_id": "query_000000",
        "repository_commit_sha": "abc",
        "task_name": "cora_node",
        "dataset_name": "cora_node",
        "split": "val",
        "runtime_query_index": 0,
        "batch_size": 1,
        "cache_mode": "memory_kv",
        "cache_tag": "tag",
        "model_configuration": {
            "hidden_size": 4,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "mem_size": 2,
            "gnn_start_layer": 26,
            "suffix_layer_ids": suffix_layers,
        },
        "query_graph_structure": {
            "num_graph_nodes": 2,
            "num_node_text_items": 3,
            "num_edge_text_items": 1,
            "num_structural_edges": 2,
            "target_index": [0],
            "question_index": [1],
            "nog_local_index": 1,
            "nog_local_indices": [1],
            "node_map": [0, 2],
            "node_map_semantics": "graph_local_node_to_encoder_text_item",
            "edge_index": [[0, 1], [1, 0]],
            "edge_map": [0, 0],
            "edge_map_semantics": "structural_edge_to_edge_text_item",
            "batch": [0, 0],
            "ptr": [0, 2],
        },
        "cache_item_inventory": inventory,
        "selective_kv_access": {
            "policy_name": "target_only",
            "eligible_item_indices": [0, 1, 3],
            "selected_key_item_indices": [0, 3],
            "selected_value_item_indices": [0],
            "complete_kv_item_indices": [0],
            "k_only_item_indices": [3],
            "v_only_item_indices": [],
            "skipped_item_indices": [1, 2],
            "key_item_mask": [True, False, False, True],
            "value_item_mask": [True, False, False, False],
            "effective_key_items_by_layer": by_key,
            "effective_value_items_by_layer": by_value,
            "selection_is_shared_across_suffix_layers": True,
        },
        "runtime_operation_shapes": {"item_order": item_order, "layers": runtime_layers},
        "traffic_metadata": {
            "byte_accounting": "quantized_data_only_excluding_scale_and_container_metadata",
            "memory_cache_bytes": 12,
            "selected_key_bytes": 24,
            "selected_value_bytes": 12,
            "full_key_bytes": 36,
            "full_value_bytes": 36,
            "edge_cache_bytes": 28,
            "nog_online_item_count": 1,
            "persistent_cache_bytes": 84,
            "runtime_loaded_cache_bytes": 48,
        },
        "summary": {
            "total_item_count": 4,
            "cacheable_item_count": 3,
            "NOG_count": 1,
            "selection_ratio_basis": "cacheable_item_layer_accesses",
            "selected_K_ratio": 2 / 3,
            "selected_V_ratio": 1 / 3,
            "selected_KV_ratio": 1 / 3,
            "persistent_cache_bytes": 84,
            "runtime_loaded_cache_bytes": 48,
        },
    }


class GOFAQueryTraceTest(unittest.TestCase):
    def test_valid_trace(self):
        self.assertEqual(validate_trace(_synthetic_trace()), [])

    def test_rejects_batch_size_two_and_bad_bytes(self):
        trace = copy.deepcopy(_synthetic_trace())
        trace["batch_size"] = 2
        trace["traffic_metadata"]["selected_key_bytes"] += 1
        errors = validate_trace(trace)
        self.assertTrue(any("batch_size must be 1" in error for error in errors))
        self.assertTrue(any("selected_key_bytes" in error for error in errors))

    def test_task_summary(self):
        summary = summarize_traces([_synthetic_trace(), _synthetic_trace()])
        task = summary["tasks"]["cora_node"]
        self.assertEqual(task["query_count"], 2)
        self.assertEqual(task["node_count"]["p95"], 2.0)
        self.assertEqual(task["persistent_cache_bytes"]["total"], 168)
        self.assertEqual(task["NOG_online_count"]["total"], 2)


if __name__ == "__main__":
    unittest.main()
