import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.large_workload_gpu_runner import (
    inspect_completed_repetitions,
    remove_repetitions,
    run_rep_transaction,
)
from scripts.large_workload_trace_state import (
    prepare_formal_trace_outputs,
    validate_resume_trace_directory,
)
from scripts.package_large_simulator_handoff import validate_handoff_artifacts
from scripts.summarize_large_workload_suite import (
    DETERMINISTIC_FIELDS,
    TIME_FIELDS,
    aggregate_per_query_medians,
    load_suite_latency_rows,
)


class FormalTraceStateTest(unittest.TestCase):
    def _manifest(self, root):
        return {
            "profile": {"samples_per_split": 2},
            "tasks": ["cora_node"],
            "paths": {"traces": str(Path(root) / "traces")},
        }

    def _write_trace(self, trace_dir, order, split, query_index, signature):
        query_id = f"query_{order:06d}"
        filename = f"{query_id}.json"
        trace = {
            "query_id": query_id,
            "task_name": "cora_node",
            "split": split,
            "runtime_query_index": query_index,
            "graph_signature": signature,
        }
        (trace_dir / filename).write_text(json.dumps(trace))
        return {
            "query_id": query_id,
            "trace_path": filename,
            "task": "cora_node",
            "split": split,
            "query_index": query_index,
            "graph_signature": signature,
        }

    def test_default_refuses_existing_and_fresh_cleans(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self._manifest(root)
            trace_dir = Path(manifest["paths"]["traces"]) / "cora_node_formal_v1"
            trace_dir.mkdir(parents=True)
            (trace_dir / "query_000000.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                prepare_formal_trace_outputs(manifest)
            state = prepare_formal_trace_outputs(manifest, fresh=True, execute=True)
            self.assertEqual(state["cora_node"]["action"], "fresh")
            self.assertEqual(list(trace_dir.iterdir()), [])

    def test_resume_requires_continuous_verified_identity(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self._manifest(root)
            trace_dir = Path(manifest["paths"]["traces"]) / "cora_node_formal_v1"
            trace_dir.mkdir(parents=True)
            entries = [
                self._write_trace(trace_dir, 0, "val", 0, "sig-0"),
                self._write_trace(trace_dir, 1, "val", 1, "sig-1"),
            ]
            with (trace_dir / "trace_index.jsonl").open("w") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry) + "\n")
            self.assertEqual(len(validate_resume_trace_directory(trace_dir, "cora_node", 2)), 2)
            state = prepare_formal_trace_outputs(manifest, resume=True)
            self.assertEqual(state["cora_node"]["existing_entries"], 2)
            entries[1]["graph_signature"] = "wrong"
            with (trace_dir / "trace_index.jsonl").open("w") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry) + "\n")
            with self.assertRaisesRegex(RuntimeError, "graph_signature mismatch"):
                validate_resume_trace_directory(trace_dir, "cora_node", 2)

    def test_resume_treats_not_started_task_as_new(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self._manifest(root)
            state = prepare_formal_trace_outputs(manifest, resume=True)
            self.assertEqual(state["cora_node"]["action"], "new")
            self.assertEqual(state["cora_node"]["existing_entries"], 0)

    def test_resume_rejects_partial_directory_without_index(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self._manifest(root)
            trace_dir = Path(manifest["paths"]["traces"]) / "cora_node_formal_v1"
            trace_dir.mkdir(parents=True)
            (trace_dir / "query_000000.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "requires an existing trace index"):
                prepare_formal_trace_outputs(manifest, resume=True)

    def test_resume_rejects_empty_index(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self._manifest(root)
            trace_dir = Path(manifest["paths"]["traces"]) / "cora_node_formal_v1"
            trace_dir.mkdir(parents=True)
            (trace_dir / "trace_index.jsonl").touch()
            with self.assertRaisesRegex(RuntimeError, "index is empty"):
                prepare_formal_trace_outputs(manifest, resume=True)


class GPUTransactionTest(unittest.TestCase):
    def test_contamination_rolls_back_csv_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path = Path(root) / "latency.csv"
            original = b"header\nverified\n"
            csv_path.write_bytes(original)
            log_path = Path(root) / "rep_001.log"

            class FakeProcess:
                pid = 99999999

                def __init__(self):
                    self.returncode = None

                def poll(self):
                    return self.returncode

                def terminate(self):
                    self.returncode = -15

                def kill(self):
                    self.returncode = -9

                def wait(self, timeout=None):
                    return self.returncode if self.returncode is not None else 0

            def popen_factory(*_args, **_kwargs):
                with csv_path.open("ab") as handle:
                    handle.write(b"contaminated\n")
                return FakeProcess()

            pid_samples = iter(([], [101, 202]))
            with self.assertRaisesRegex(RuntimeError, "contaminated"):
                run_rep_transaction(
                    ["fake-command"],
                    csv_path,
                    log_path,
                    verify_complete=lambda: None,
                    query_pids=lambda: next(pid_samples),
                    popen_factory=popen_factory,
                    sleep=lambda _seconds: None,
                )
            self.assertEqual(csv_path.read_bytes(), original)
            self.assertFalse(log_path.exists())
            archived_logs = list(Path(root).glob("rep_001.contaminated_*.log"))
            self.assertEqual(len(archived_logs), 1)
            self.assertIn("CONTAMINATED", archived_logs[0].read_text())

            class SuccessfulProcess:
                pid = 99999998

                def poll(self):
                    return 0

                def wait(self, timeout=None):
                    return 0

            run_rep_transaction(
                ["successful-command"],
                csv_path,
                log_path,
                verify_complete=lambda: None,
                query_pids=lambda: [],
                popen_factory=lambda *_args, **_kwargs: SuccessfulProcess(),
                sleep=lambda _seconds: None,
            )
            self.assertTrue(log_path.exists())
            self.assertEqual(len(list(Path(root).glob("rep_001.contaminated_*.log"))), 1)

    def _write_latency_fixture(self, root):
        trace_dir = Path(root) / "trace"
        trace_dir.mkdir()
        entries = []
        for order, split, query_index in ((0, "val", 0), (1, "test", 0)):
            filename = f"query_{order:06d}.json"
            trace = {
                "task_name": "cora_node",
                "split": split,
                "runtime_query_index": query_index,
                "query_uid": f"uid-{order}",
                "graph_signature": f"sig-{order}",
                "workload_profile": {"name": "profile"},
            }
            (trace_dir / filename).write_text(json.dumps(trace))
            entries.append({
                "task": "cora_node",
                "split": split,
                "query_index": query_index,
                "query_uid": f"uid-{order}",
                "graph_signature": f"sig-{order}",
                "workload_profile": "profile",
                "trace_path": filename,
            })
        index_path = trace_dir / "trace_index.jsonl"
        with index_path.open("w") as handle:
            for entry in entries:
                handle.write(json.dumps(entry) + "\n")
        csv_path = Path(root) / "latency.csv"
        fields = [
            "task", "profile_mode", "rep", "split", "trace_order", "query_index",
            "query_uid", "graph_signature", "workload_profile", "cache_misses", "fallback_count",
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for rep in (0, 1):
                for order, entry in enumerate(entries):
                    writer.writerow({
                        "task": "cora_node",
                        "profile_mode": "cache_w8a8_m4k2v2",
                        "rep": rep,
                        "split": entry["split"],
                        "trace_order": order,
                        "query_index": entry["query_index"],
                        "query_uid": entry["query_uid"],
                        "graph_signature": entry["graph_signature"],
                        "workload_profile": "profile",
                        "cache_misses": 0,
                        "fallback_count": 0,
                    })
            writer.writerow({
                "task": "cora_node", "profile_mode": "cache_w8a8_m4k2v2", "rep": 2,
                "split": "val", "trace_order": 0, "query_index": 0, "query_uid": "uid-0",
                "graph_signature": "sig-0", "workload_profile": "profile",
                "cache_misses": 0, "fallback_count": 0,
            })
        return csv_path, index_path

    def test_repetition_resume_detects_complete_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path, index_path = self._write_latency_fixture(root)
            complete, incomplete = inspect_completed_repetitions(
                csv_path, index_path, "cora_node", "cache_w8a8_m4k2v2", 1
            )
            self.assertEqual(complete, {0, 1})
            self.assertEqual(incomplete, {2})
            remove_repetitions(csv_path, incomplete)
            complete, incomplete = inspect_completed_repetitions(
                csv_path, index_path, "cora_node", "cache_w8a8_m4k2v2", 1
            )
            self.assertEqual(complete, {0, 1})
            self.assertEqual(incomplete, set())


class MedianSummaryTest(unittest.TestCase):
    def test_per_query_median_does_not_treat_reps_as_queries(self):
        rows = []
        for query_index in (0, 1):
            for rep, value in enumerate((1.0, 100.0, 3.0)):
                row = {
                    "task": "cora_node",
                    "profile_mode": "cache_w8a8_m4k2v2",
                    "split": "val",
                    "query_uid": f"uid-{query_index}",
                    "trace_order": str(query_index),
                    "query_index": str(query_index),
                    "workload_profile": "profile",
                    "graph_signature": f"sig-{query_index}",
                    "rep": str(rep),
                }
                row.update({field: str(value + query_index) for field in TIME_FIELDS})
                row.update({field: str(query_index + 10) for field in DETERMINISTIC_FIELDS})
                rows.append(row)
        medians = aggregate_per_query_medians(rows, expected_reps=3)
        self.assertEqual(len(medians), 2)
        self.assertEqual(medians[0]["query_wall_ms"], 3.0)
        self.assertEqual(medians[1]["query_wall_ms"], 4.0)
        self.assertEqual(medians[0]["logical_memory_loaded_bytes"], 10)

    def test_rejects_nondeterministic_counter_across_reps(self):
        rows = []
        for rep in range(3):
            row = {
                "task": "cora_node",
                "profile_mode": "cache_w8a8_m4k2v2",
                "split": "val",
                "query_uid": "uid-0",
                "trace_order": "0",
                "query_index": "0",
                "workload_profile": "profile",
                "graph_signature": "sig-0",
                "rep": str(rep),
            }
            row.update({field: "1.0" for field in TIME_FIELDS})
            row.update({field: "10" for field in DETERMINISTIC_FIELDS})
            row["cache_hits"] = str(10 + rep)
            rows.append(row)
        with self.assertRaisesRegex(RuntimeError, "cache_hits differs"):
            aggregate_per_query_medians(rows, expected_reps=3)

    def test_summary_requires_all_mode_task_csvs_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            suite = {
                "profile": {"samples_per_split": 1},
                "reps": 3,
                "tasks": ["cora_node"],
                "paths": {"latency": str(Path(root) / "latency")},
            }
            with self.assertRaisesRegex(RuntimeError, "requires every mode/task"):
                load_suite_latency_rows(suite, allow_partial=False)
            rows, counts = load_suite_latency_rows(suite, allow_partial=True)
            self.assertEqual(rows, [])
            self.assertEqual(counts, {})


class SimulatorHandoffValidationTest(unittest.TestCase):
    def test_rejects_missing_handoff_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            suite = {
                "profile": {"samples_per_split": 1},
                "reps": 1,
                "tasks": ["cora_node"],
                "paths": {
                    "root": str(root_path),
                    "manifests": str(root_path / "manifests"),
                    "traces": str(root_path / "traces"),
                    "latency": str(root_path / "latency"),
                    "summary": str(root_path / "summary"),
                },
            }
            with self.assertRaisesRegex(RuntimeError, "expected 2 trace JSON files"):
                validate_handoff_artifacts(suite)


if __name__ == "__main__":
    unittest.main()
