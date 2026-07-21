import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

from scripts.validate_h100_per_query_latency import REQUIRED_FIELDS, load_csv_rows, validate_rows


GOFA_MODULE_DIR = Path(__file__).resolve().parents[1] / "modules" / "gofa"


def _load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_latency_modules():
    package_name = "_gofa_latency_test_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(GOFA_MODULE_DIR)]
    sys.modules[package_name] = package
    accumulator = _load_module(
        f"{package_name}.latency_event_accumulator",
        GOFA_MODULE_DIR / "latency_event_accumulator.py",
    )
    _load_module(f"{package_name}.query_trace", GOFA_MODULE_DIR / "query_trace.py")

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        exporter = _load_module(
            f"{package_name}.per_query_latency",
            GOFA_MODULE_DIR / "per_query_latency.py",
        )
    finally:
        if previous_torch is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = previous_torch
    return accumulator, exporter


LATENCY_ACCUMULATOR, PER_QUERY_LATENCY = _load_latency_modules()


class FakeEvent:
    def __init__(self, timestamp):
        self.timestamp = float(timestamp)

    def elapsed_time(self, end):
        return end.timestamp - self.timestamp


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
                    "suffix_transformer_gpu_ms": 10.0,
                    "quant_kv_attention_gpu_ms": 5.0,
                    "suffix_gnn_gpu_ms": 8.0,
                    "gnn_score_gpu_ms": 2.0,
                    "gnn_message_gpu_ms": 2.0,
                    "gnn_update_gpu_ms": 2.0,
                    "gnn_other_gpu_ms": 2.0,
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

    def test_rejects_detail_time_outside_parent_region(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path, trace_template = self._write_fixture(root)
            rows = load_csv_rows([csv_path])
            rows[0]["quant_kv_attention_gpu_ms"] = "12.0"
            with self.assertRaisesRegex(RuntimeError, "exceeds suffix_transformer"):
                validate_rows(rows, trace_template, expected_per_split=2)


class LatencyEventLifecycleTest(unittest.TestCase):
    def test_disabled_detail_capture_tracks_no_events(self):
        accumulator = LATENCY_ACCUMULATOR.LatencyEventAccumulator()
        accumulator.begin(enabled=False)

        token = accumulator.start("int_qk", FakeEvent(0.0))
        events = accumulator.finish()

        self.assertIsNone(token)
        self.assertTrue(all(not pairs for pairs in events.values()))
        self.assertEqual(accumulator.pending_count(), 0)
        self.assertEqual(accumulator.pair_count(), 0)

    def test_accumulates_multiple_layers_and_calls(self):
        accumulator = LATENCY_ACCUMULATOR.LatencyEventAccumulator()
        accumulator.begin(enabled=True)
        for start, end in ((0.0, 1.25), (2.0, 4.5), (5.0, 5.75)):
            token = accumulator.start("int_qk", FakeEvent(start))
            accumulator.end(token, FakeEvent(end))
        token = accumulator.start("int_pv", FakeEvent(10.0))
        accumulator.end(token, FakeEvent(13.0))

        events = accumulator.finish()

        self.assertAlmostEqual(LATENCY_ACCUMULATOR.elapsed_event_pairs_ms(events["int_qk"]), 4.5)
        self.assertAlmostEqual(LATENCY_ACCUMULATOR.elapsed_event_pairs_ms(events["int_pv"]), 3.0)
        self.assertEqual(accumulator.pending_count(), 0)
        self.assertEqual(accumulator.pair_count(), 0)

    def test_abort_query_clears_unfinished_events(self):
        accumulator = LATENCY_ACCUMULATOR.LatencyEventAccumulator()
        accumulator.begin(enabled=True)
        accumulator.start("quant_kv_attention", FakeEvent(0.0))

        class FakeBaseModel:
            def __init__(self):
                self.abort_calls = 0

            def abort_per_query_latency_capture(self):
                self.abort_calls += 1
                accumulator.abort()

        base_model = FakeBaseModel()

        class FakeICAE:
            def get_base_model(self):
                return types.SimpleNamespace(model=base_model)

        owner = types.SimpleNamespace(model=types.SimpleNamespace(icae=FakeICAE()))
        exporter = PER_QUERY_LATENCY.GOFAPerQueryLatencyExporter(owner, {}, enabled=False)
        exporter.current = {"active": True}

        exporter.abort_query()

        self.assertEqual(base_model.abort_calls, 1)
        self.assertEqual(accumulator.pending_count(), 0)
        self.assertEqual(accumulator.pair_count(), 0)
        self.assertIsNone(exporter.current)

    def test_warmup_query_does_not_write_csv(self):
        with tempfile.TemporaryDirectory() as root:
            output_csv = os.path.join(root, "latency.csv")
            exporter = PER_QUERY_LATENCY.GOFAPerQueryLatencyExporter(
                None,
                {"output_csv": output_csv, "trace_index_path": "unused", "append": False},
                enabled=True,
            )
            exporter._initialize_csv()
            exporter.set_context("cora_node", "val", 0, is_warmup=True)

            self.assertFalse(exporter.begin_query(graph=None, device=None))
            with open(output_csv, newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])

    def test_append_starts_next_repetition_without_rewriting_rows(self):
        with tempfile.TemporaryDirectory() as root:
            output_csv = os.path.join(root, "latency.csv")
            with open(output_csv, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=PER_QUERY_LATENCY.CSV_FIELDS)
                writer.writeheader()
                row = {field: 0 for field in PER_QUERY_LATENCY.CSV_FIELDS}
                row.update({"task": "cora_node", "split": "val", "rep": 0})
                writer.writerow(row)
            original_size = os.path.getsize(output_csv)

            exporter = PER_QUERY_LATENCY.GOFAPerQueryLatencyExporter(
                None,
                {"output_csv": output_csv, "trace_index_path": "unused", "append": True},
                enabled=True,
            )
            exporter._initialize_csv()
            exporter.set_context("cora_node", "val", 0, is_warmup=False)

            self.assertEqual(exporter.context["rep"], 1)
            self.assertEqual(os.path.getsize(output_csv), original_size)


if __name__ == "__main__":
    unittest.main()
