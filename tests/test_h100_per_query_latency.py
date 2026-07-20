import csv
import json
import os
import tempfile
import unittest

from scripts.validate_h100_per_query_latency import REQUIRED_FIELDS, load_csv_rows, validate_rows


class H100PerQueryLatencyValidatorTest(unittest.TestCase):
    def _write_fixture(self, root):
        trace_dir = os.path.join(root, "cora_node_formal_v1")
        os.makedirs(trace_dir)
        index_path = os.path.join(trace_dir, "trace_index.jsonl")
        traces = []
        for order, split, query_index in (
            (0, "val", 0),
            (1, "val", 1),
            (2, "test", 0),
            (3, "test", 1),
        ):
            query_id = f"query_{order:06d}"
            filename = f"{query_id}.json"
            with open(os.path.join(trace_dir, filename), "w") as handle:
                json.dump({
                    "query_id": query_id,
                    "task_name": "cora_node",
                    "split": split,
                    "runtime_query_index": query_index,
                    "cache_item_inventory": [{"cache_key": f"key-{order}"}],
                }, handle)
            traces.append({
                "query_id": query_id,
                "task": "cora_node",
                "split": split,
                "trace_path": filename,
            })
        with open(index_path, "w") as handle:
            for entry in traces:
                handle.write(json.dumps(entry) + "\n")

        csv_path = os.path.join(root, "latency.csv")
        fields = sorted(REQUIRED_FIELDS)
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for order, split, query_index in (
                (0, "val", 0),
                (1, "val", 1),
                (2, "test", 0),
                (3, "test", 1),
            ):
                row = {field: 0 for field in fields}
                row.update({
                    "task": "cora_node",
                    "split": split,
                    "trace_order": order,
                    "query_index": query_index,
                    "query_uid": f"query_{order:06d}",
                    "rep": 0,
                    "quant_kv_attention_calls": 6,
                })
                writer.writerow(row)
        return csv_path, os.path.join(root, "${TASK}_formal_v1", "trace_index.jsonl")

    def test_accepts_complete_trace_aligned_rep(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path, trace_template = self._write_fixture(root)
            summaries = validate_rows(load_csv_rows([csv_path]), trace_template, expected_per_split=2)
            self.assertEqual(summaries, [("cora_node", 0, 4)])

    def test_rejects_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path, trace_template = self._write_fixture(root)
            rows = load_csv_rows([csv_path])
            rows[0]["fallback_count"] = "1"
            with self.assertRaisesRegex(RuntimeError, "fallback_count must be zero"):
                validate_rows(rows, trace_template, expected_per_split=2)


if __name__ == "__main__":
    unittest.main()
