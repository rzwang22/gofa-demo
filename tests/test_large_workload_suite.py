import json
import os
from pathlib import Path
import tempfile
import types
import unittest

try:
    import torch
except ImportError:
    torch = None

from modules.gofa.int_gemm_backend import execute_int_mm, weight_qmax
from modules.gofa.workload_profile import (
    graph_signature,
    resolve_saved_workload_names,
    validate_runtime_sampling,
)
from scripts.large_workload_common import TASK_WAYS, build_task_configs, normalize_args, prepare_suite
from tests.test_h100_per_query_latency import PER_QUERY_LATENCY


class LargeWorkloadProfileTest(unittest.TestCase):
    def _args(self, root):
        runtime_root = Path(root) / "runtime_inputs"
        data_root = runtime_root / "data"
        model_root = runtime_root / "model"
        checkpoint_root = runtime_root / "checkpoints"
        for path in (data_root, model_root, checkpoint_root):
            path.mkdir(parents=True, exist_ok=True)
        load_path = checkpoint_root / "instruct_2_ckpt.pth"
        load_path.touch()
        return types.SimpleNamespace(
            profile_root=root,
            profile_name="large_h6_n32_s100",
            hops=6,
            max_nodes_per_hop=32,
            samples_per_split=100,
            seed=1,
            tasks=["cora_node"],
            reps=3,
            data_root=str(data_root),
            model_name_or_path=str(model_root),
            checkpoint_dir=str(checkpoint_root),
            load_dir=str(load_path),
            kv_policy="target_1hop",
            kv_target_hops=1,
            overwrite_suite=False,
        )

    def test_h6_n32_runtime_sampling_is_accepted(self):
        profile, tasks = normalize_args(self._args("/tmp/profile"))
        errors = validate_runtime_sampling(
            profile,
            seed=1,
            eval_sample_size=100,
            tasks=tasks,
            sample_sizes=[100],
            hops=[6],
            max_nodes=[32],
        )
        self.assertEqual(errors, [])

    def test_saved_workload_name_and_signature_are_stable(self):
        profile, tasks = normalize_args(self._args("/tmp/profile"))
        names = resolve_saved_workload_names(profile, tasks, "val", {})
        self.assertEqual(
            names,
            ["large_h6_n32_s100__cora_node__val__s1__h6__n32"],
        )
        components = {
            "node_map": [0, 2, 1],
            "edge_map": [0, 1],
            "edge_index": [[0, 1], [1, 2]],
            "target_index": [0],
            "question_index": [2],
        }
        before = graph_signature("cora_node", "val", 0, **components)
        reloaded = json.loads(json.dumps(components))
        after = graph_signature("cora_node", "val", 0, **reloaded)
        self.assertEqual(before, after)

    def test_three_modes_share_formal_trace_and_saved_workload(self):
        with tempfile.TemporaryDirectory() as root:
            args = self._args(root)
            profile, _ = normalize_args(args)
            configs = build_task_configs(args, profile, "cora_node")
            trace_paths = {
                configs[f"gpu_{mode}"]["gofa_per_query_latency"]["trace_index_path"]
                for mode in ("nocache_bf16", "cache_bf16", "cache_w8a8_m4k2v2")
            }
            self.assertEqual(len(trace_paths), 1)
            for mode in ("nocache_bf16", "cache_bf16", "cache_w8a8_m4k2v2"):
                config = configs[f"gpu_{mode}"]
                self.assertTrue(config["inf_from_saved"])
                self.assertEqual(config["workload_profile"], profile)

    def test_task_ways_runtime_paths_and_kv_policy_are_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            args = self._args(root)
            profile, _ = normalize_args(args)
            for task, ways in TASK_WAYS.items():
                configs = build_task_configs(args, profile, task)
                for config in configs.values():
                    self.assertEqual(config["task_names"], [task])
                    self.assertEqual(config["train_task_names"], [task])
                    self.assertEqual(config["eval_task_names"], [task])
                    self.assertEqual(config["ways"], ways)
                    self.assertEqual(config["inf_ways"], [ways])
                    self.assertEqual(config["sample_size_per_task"], 100)
                    self.assertEqual(config["train_sample_size"], -1)
                    self.assertTrue(config["load_model"])
                    self.assertTrue(config["load_dir"])
                    self.assertTrue(config["data_root_path"])

                formal_quant = configs["formal_trace"]["scheme_b_quant"]
                gpu_quant = configs["gpu_cache_w8a8_m4k2v2"]["scheme_b_quant"]
                self.assertEqual(formal_quant["kv_base_load_policy"], "target_1hop")
                self.assertEqual(formal_quant["kv_base_target_hops"], 1)
                self.assertEqual(
                    (formal_quant["kv_base_load_policy"], formal_quant["kv_base_target_hops"]),
                    (gpu_quant["kv_base_load_policy"], gpu_quant["kv_base_target_hops"]),
                )

    def test_required_runtime_path_types_are_checked(self):
        with tempfile.TemporaryDirectory() as root:
            args = self._args(root)
            args.load_dir = args.checkpoint_dir
            with self.assertRaisesRegex(ValueError, "load-dir must be an existing file"):
                normalize_args(args)

    def test_suite_identity_mismatch_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            args = self._args(root)
            prepare_suite(args)
            args.reps = 2
            with self.assertRaisesRegex(RuntimeError, "suite identity mismatch"):
                prepare_suite(args)


class ProfileModeAndIntGemmTest(unittest.TestCase):
    def _base_model(self):
        return types.SimpleNamespace(
            gnn_start_layer=26,
            config=types.SimpleNamespace(num_hidden_layers=32),
        )

    def _owner(self, encoder_cache_enabled=False, encoder_cache_mode="memory_kv"):
        base_model = self._base_model()

        class ICAE:
            def get_base_model(self):
                return types.SimpleNamespace(model=base_model)

        return types.SimpleNamespace(
            encoder_cache_enabled=encoder_cache_enabled,
            encoder_cache_mode=encoder_cache_mode,
            model_args=types.SimpleNamespace(
                encoder_cache_skip_nog=encoder_cache_enabled,
                workload_profile={
                    "name": "large_h6_n32_s100",
                    "seed": 1,
                    "samples_per_split": 100,
                    "hops": 6,
                    "max_nodes_per_hop": 32,
                },
            ),
            scheme_b_quant_enabled=False,
            scheme_b_quant={},
            scheme_b_int_gemm_enabled=False,
            scheme_b_int_gemm={},
            scheme_b_quant_kv_attention_enabled=False,
            scheme_b_quant_kv_attention={},
            gofa_query_trace_enabled=False,
            model=types.SimpleNamespace(icae=ICAE()),
        )

    def test_nocache_exporter_setup_does_not_require_cache(self):
        with tempfile.TemporaryDirectory() as root:
            exporter = PER_QUERY_LATENCY.GOFAPerQueryLatencyExporter(
                self._owner(encoder_cache_enabled=False),
                {
                    "profile_mode": "nocache_bf16",
                    "output_csv": os.path.join(root, "latency.csv"),
                    "trace_index_path": os.path.join(root, "trace_index.jsonl"),
                    "export_gpu_time": False,
                    "export_detail_gpu_time": False,
                },
                enabled=True,
            )
            exporter.validate_setup()
            self.assertTrue(os.path.isfile(exporter.config["output_csv"]))

    def test_w8_qmax_and_backend_execution(self):
        calls = []

        def fake_int_mm(lhs, rhs):
            calls.append((lhs, rhs))
            return "int32-output"

        self.assertEqual(weight_qmax(8), 127)
        self.assertEqual(execute_int_mm(fake_int_mm, "x-int8", "w-int8"), "int32-output")
        self.assertEqual(calls, [("x-int8", "w-int8")])

    @unittest.skipUnless(
        torch is not None and hasattr(torch, "_int_mm") and torch.cuda.is_available(),
        "CUDA torch._int_mm is unavailable in the CPU test environment",
    )
    def test_w8a8_executes_real_torch_int_mm_on_cuda(self):
        lhs = torch.randint(-127, 128, (32, 64), dtype=torch.int8, device="cuda").contiguous()
        rhs = torch.randint(-127, 128, (64, 32), dtype=torch.int8, device="cuda").contiguous()
        output = execute_int_mm(torch._int_mm, lhs, rhs)
        self.assertEqual(tuple(output.shape), (32, 32))
        self.assertEqual(output.dtype, torch.int32)

    def test_append_repetitions_advance_zero_one_two(self):
        exporter = PER_QUERY_LATENCY.GOFAPerQueryLatencyExporter(None, {}, enabled=False)
        observed = []
        for _rep in range(3):
            exporter.set_context("cora_node", "val", 0)
            observed.append(exporter.context["rep"])
            exporter.set_context("cora_node", "val", 1)
        self.assertEqual(observed, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
