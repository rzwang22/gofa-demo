import csv
import json
import os
import time

import torch

from .latency_event_accumulator import elapsed_event_pairs_ms
from .query_trace import FORMAL_BATCH_ERROR, infer_graph_batch_size
from .workload_profile import graph_signature, normalize_workload_profile, query_uid


PROFILE_MODES = {
    "nocache_bf16",
    "cache_bf16",
    "cache_w8a8_m4k2v2",
    "cache_w4a8_m4k2v2",
}
CACHE_PROFILE_MODES = {"cache_bf16", "cache_w8a8_m4k2v2", "cache_w4a8_m4k2v2"}
QUANT_PROFILE_MODES = {"cache_w8a8_m4k2v2", "cache_w4a8_m4k2v2"}


CSV_FIELDS = [
    "task",
    "split",
    "trace_order",
    "query_index",
    "query_uid",
    "rep",
    "query_wall_ms",
    "query_gpu_ms",
    "encoder_wall_ms",
    "decoder_wall_ms",
    "cache_load_wall_ms",
    "online_nog_prefix_wall_ms",
    "cache_assembly_wall_ms",
    "suffix_wall_ms",
    "suffix_gnn_gpu_ms",
    "suffix_transformer_gpu_ms",
    "cache_hits",
    "cache_misses",
    "cache_skips",
    "quant_kv_attention_calls",
    "fallback_count",
    "quant_kv_attention_gpu_ms",
    "kv_prepare_gpu_ms",
    "int_qk_gpu_ms",
    "softmax_prob_quant_gpu_ms",
    "int_pv_gpu_ms",
    "gnn_score_gpu_ms",
    "gnn_message_gpu_ms",
    "gnn_update_gpu_ms",
    "gnn_other_gpu_ms",
    "profile_mode",
    "workload_profile",
    "graph_signature",
    "prefix_transformer_gpu_ms",
    "dense_fc_gpu_ms",
    "attention_gpu_ms",
    "norm_residual_other_gpu_ms",
    "logical_memory_loaded_bytes",
    "logical_key_loaded_bytes",
    "logical_value_loaded_bytes",
    "int_gemm_call_count",
]


def normalize_trace_split(value):
    value = str(value or "").strip().lower()
    if value in {"val", "valid", "validation"}:
        return "val"
    if value in {"test", "testing"}:
        return "test"
    return value


def resolve_task_path(path_template, task):
    path = str(path_template or "")
    path = path.replace("${TASK}", str(task)).replace("$TASK", str(task))
    path = path.replace("{TASK}", str(task)).replace("{task}", str(task))
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _read_json(path):
    with open(path) as handle:
        return json.load(handle)


def load_trace_index_catalog(path, expected_task=None, strict=True):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise RuntimeError(f"GOFA per-query latency trace index does not exist: {path}")

    entries = []
    by_query = {}
    with open(path) as handle:
        for trace_order, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                index_entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in GOFA trace index {path} at line {trace_order + 1}: {exc}"
                ) from exc

            trace_path = index_entry.get("trace_path")
            if not trace_path:
                raise RuntimeError(
                    f"GOFA trace index entry {trace_order} has no trace_path: {path}"
                )
            if not os.path.isabs(trace_path):
                trace_path = os.path.join(os.path.dirname(path), trace_path)
            if not os.path.isfile(trace_path):
                raise RuntimeError(
                    f"GOFA trace file referenced by index does not exist: {trace_path}"
                )
            trace = _read_json(trace_path)

            index_task = index_entry.get("task")
            trace_task = trace.get("task_name")
            task = trace_task or index_task
            index_split = normalize_trace_split(index_entry.get("split"))
            trace_split = normalize_trace_split(trace.get("split"))
            split = trace_split or index_split
            query_index = trace.get("runtime_query_index", index_entry.get("query_index"))
            if query_index is None:
                raise RuntimeError(
                    f"GOFA trace entry {trace_order} has no runtime_query_index: {trace_path}"
                )
            query_index = int(query_index)

            if strict and index_task is not None and trace_task is not None and index_task != trace_task:
                raise RuntimeError(
                    "GOFA trace index task mismatch: "
                    f"trace_order={trace_order}, index_task={index_task}, trace_task={trace_task}"
                )
            if strict and index_split and trace_split and index_split != trace_split:
                raise RuntimeError(
                    "GOFA trace index split mismatch: "
                    f"trace_order={trace_order}, index_split={index_split}, trace_split={trace_split}"
                )
            if strict and expected_task is not None and task != expected_task:
                raise RuntimeError(
                    "GOFA trace index contains an unexpected task: "
                    f"trace_order={trace_order}, expected={expected_task}, actual={task}"
                )
            if strict and int(trace.get("batch_size", 1)) != 1:
                raise RuntimeError(
                    f"GOFA trace entry {trace_order} has batch_size={trace.get('batch_size')}, expected 1."
                )
            index_query_index = index_entry.get("query_index")
            if strict and index_query_index is not None and int(index_query_index) != query_index:
                raise RuntimeError(
                    "GOFA trace index query_index mismatch: "
                    f"trace_order={trace_order}, index={index_query_index}, trace={query_index}"
                )

            inventory = trace.get("cache_item_inventory")
            if strict and not isinstance(inventory, list):
                raise RuntimeError(
                    f"GOFA trace {trace_path} has no cache_item_inventory for strict matching."
                )
            cache_keys = []
            for item in inventory or []:
                cache_key = item.get("cache_key") if isinstance(item, dict) else None
                if strict and not cache_key:
                    raise RuntimeError(
                        f"GOFA trace {trace_path} contains an inventory item without cache_key."
                    )
                cache_keys.append(cache_key)
            index_cache_keys = index_entry.get("cache_keys")
            if strict and index_cache_keys is not None and list(index_cache_keys) != cache_keys:
                raise RuntimeError(
                    f"GOFA trace index cache_keys mismatch at trace_order={trace_order}."
                )

            index_query_uid = index_entry.get("query_uid") or index_entry.get("query_id")
            trace_query_uid = trace.get("query_uid") or trace.get("query_id")
            if strict and index_query_uid and trace_query_uid and str(index_query_uid) != str(trace_query_uid):
                raise RuntimeError(
                    "GOFA trace index query UID mismatch: "
                    f"trace_order={trace_order}, index={index_query_uid}, trace={trace_query_uid}"
                )
            query_uid = index_query_uid or trace_query_uid
            if strict and not query_uid:
                raise RuntimeError(
                    f"GOFA trace entry {trace_order} has neither query_uid nor query_id."
                )
            index_signature = index_entry.get("graph_signature")
            trace_signature = trace.get("graph_signature")
            if strict and index_signature and trace_signature and index_signature != trace_signature:
                raise RuntimeError(
                    f"GOFA trace index graph_signature mismatch at trace_order={trace_order}."
                )
            signature = trace_signature or index_signature
            if strict and not signature:
                raise RuntimeError(f"GOFA trace entry {trace_order} has no graph_signature.")
            workload = trace.get("workload_profile") or index_entry.get("workload_profile")
            if isinstance(workload, dict):
                workload = workload.get("name")
            traffic = trace.get("traffic_metadata") or {}
            entry = {
                "task": task,
                "split": split,
                "trace_order": int(trace_order),
                "query_index": query_index,
                "query_uid": str(query_uid or f"trace_{trace_order:06d}"),
                "graph_signature": str(signature or ""),
                "workload_profile": str(workload or ""),
                "cache_keys": cache_keys,
                "logical_memory_loaded_bytes": int(traffic.get("memory_cache_bytes", 0)),
                "logical_key_loaded_bytes": int(traffic.get("selected_key_bytes", 0)),
                "logical_value_loaded_bytes": int(traffic.get("selected_value_bytes", 0)),
                "trace_path": os.path.abspath(trace_path),
            }
            key = (split, query_index)
            if key in by_query:
                raise RuntimeError(
                    "GOFA trace index contains duplicate split/query indices: "
                    f"split={split}, query_index={query_index}, path={path}"
                )
            by_query[key] = entry
            entries.append(entry)

    if not entries:
        raise RuntimeError(f"GOFA per-query latency trace index is empty: {path}")
    return {"path": path, "entries": entries, "by_query": by_query}


class GOFAPerQueryLatencyExporter:
    def __init__(self, owner, config, enabled=True):
        self.owner = owner
        self.config = dict(config)
        self.profile_mode = str(self.config.get("profile_mode", "cache_w8a8_m4k2v2"))
        self.enabled = bool(enabled)
        self.context = {}
        self.current = None
        self.catalogs = {}
        self.rep_by_split = {}
        self.last_query_index = {}
        self.last_trace_order = {}
        self.existing_next_rep = {}
        self.csv_initialized = False
        if not self.enabled:
            return
        if self.profile_mode not in PROFILE_MODES:
            raise ValueError(
                f"Unsupported gofa_per_query_latency.profile_mode={self.profile_mode!r}; "
                f"expected one of {sorted(PROFILE_MODES)}."
            )

        output_csv = self.config.get("output_csv", "")
        if not output_csv:
            raise ValueError("gofa_per_query_latency.output_csv must be set when enabled.")
        if not self.config.get("trace_index_path", ""):
            raise ValueError("gofa_per_query_latency.trace_index_path must be set when enabled.")
        output_csv = os.path.abspath(os.path.expanduser(os.path.expandvars(output_csv)))
        self.config["output_csv"] = output_csv
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    def _initialize_csv(self):
        path = self.config["output_csv"]
        append = bool(self.config.get("append", False))
        if append and os.path.isfile(path) and os.path.getsize(path) > 0:
            with open(path, newline="") as handle:
                reader = csv.DictReader(handle)
                if list(reader.fieldnames or []) != CSV_FIELDS:
                    raise RuntimeError(
                        "Cannot append GOFA per-query latency rows to a CSV with a different schema: "
                        f"{path}"
                    )
                for row in reader:
                    task = row.get("task")
                    split = normalize_trace_split(row.get("split"))
                    try:
                        rep = int(row.get("rep", 0))
                    except (TypeError, ValueError):
                        continue
                    key = (task, split)
                    self.existing_next_rep[key] = max(self.existing_next_rep.get(key, 0), rep + 1)
            return
        with open(path, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()

    def validate_setup(self):
        if not self.enabled:
            return
        owner = self.owner
        errors = []
        requires_cache = self.profile_mode in CACHE_PROFILE_MODES
        quant_mode = self.profile_mode in QUANT_PROFILE_MODES
        if requires_cache:
            if not bool(owner.encoder_cache_enabled) or owner.encoder_cache_mode != "memory_kv":
                errors.append("cache profile modes require use_encoder_cache=True and encoder_cache_mode=memory_kv")
            if not bool(getattr(owner.model_args, "encoder_cache_skip_nog", False)):
                errors.append("cache profile modes require encoder_cache_skip_nog=True")
        elif bool(owner.encoder_cache_enabled):
            errors.append("nocache_bf16 requires use_encoder_cache=False")

        if quant_mode and not bool(owner.scheme_b_quant_enabled):
            errors.append("quant profile mode requires scheme_b_quant.enabled=True")
        if quant_mode and bool(owner.scheme_b_quant_enabled):
            for key, expected in (("memory_base_bits", 4), ("key_base_bits", 2), ("value_base_bits", 2)):
                if int(owner.scheme_b_quant.get(key, -1)) != expected:
                    errors.append(f"scheme_b_quant.{key} must be {expected}")
            if not bool(owner.scheme_b_quant.get("strict", False)):
                errors.append("scheme_b_quant.strict=True is required")
            for key in ("load_memory_delta", "load_key_delta", "load_value_delta"):
                if bool(owner.scheme_b_quant.get(key, False)):
                    errors.append(f"scheme_b_quant.{key} must be False for canonical base-only M4K2V2")
            if not bool(owner.scheme_b_quant.get("load_key_base", False)):
                errors.append("scheme_b_quant.load_key_base=True is required")
            if not bool(owner.scheme_b_quant.get("load_value_base", False)):
                errors.append("scheme_b_quant.load_value_base=True is required")
        if not quant_mode and bool(owner.scheme_b_quant_enabled):
            errors.append(f"{self.profile_mode} requires scheme_b_quant.enabled=False")

        int_gemm = owner.scheme_b_int_gemm
        if quant_mode and not bool(owner.scheme_b_int_gemm_enabled):
            errors.append("quant profile mode requires scheme_b_int_gemm.enabled=True")
        if quant_mode and bool(owner.scheme_b_int_gemm_enabled):
            expected_weight_bits = 8 if self.profile_mode == "cache_w8a8_m4k2v2" else 4
            if (
                int(int_gemm.get("weight_bits", -1)) != expected_weight_bits
                or int(int_gemm.get("activation_bits", -1)) != 8
            ):
                errors.append(f"scheme_b_int_gemm must use W{expected_weight_bits}A8")
            if str(int_gemm.get("backend")) != "torch_int_mm":
                errors.append("scheme_b_int_gemm.backend must be torch_int_mm")
            if not bool(int_gemm.get("quantize_attention", False)):
                errors.append("scheme_b_int_gemm.quantize_attention=True is required")
            if not bool(int_gemm.get("quantize_mlp", False)):
                errors.append("scheme_b_int_gemm.quantize_mlp=True is required")
            if bool(int_gemm.get("fallback_to_fake_quant", True)):
                errors.append("scheme_b_int_gemm.fallback_to_fake_quant must be False")
        if not quant_mode and bool(owner.scheme_b_int_gemm_enabled):
            errors.append(f"{self.profile_mode} requires scheme_b_int_gemm.enabled=False")

        kv_attention = owner.scheme_b_quant_kv_attention
        if quant_mode and not bool(owner.scheme_b_quant_kv_attention_enabled):
            errors.append("quant profile mode requires scheme_b_quant_kv_attention.enabled=True")
        if quant_mode and bool(owner.scheme_b_quant_kv_attention_enabled):
            if int(kv_attention.get("key_bits", -1)) != 2 or int(kv_attention.get("value_bits", -1)) != 2:
                errors.append("scheme_b_quant_kv_attention must use K2V2")
            if str(kv_attention.get("backend")) != "torch_int_mm_qscale_fold":
                errors.append("scheme_b_quant_kv_attention.backend must be torch_int_mm_qscale_fold")
            if not bool(kv_attention.get("key_scale_fold_into_q", False)):
                errors.append("scheme_b_quant_kv_attention.key_scale_fold_into_q=True is required")
            if int(kv_attention.get("quantize_query_bits", -1)) != 8:
                errors.append("scheme_b_quant_kv_attention.quantize_query_bits must be 8")
            if not bool(kv_attention.get("use_int_qk", False)):
                errors.append("scheme_b_quant_kv_attention.use_int_qk=True is required")
            if str(kv_attention.get("pv_compute_mode")) != "int_pv":
                errors.append("scheme_b_quant_kv_attention.pv_compute_mode must be int_pv")
            if int(kv_attention.get("quantize_prob_bits", -1)) != 8:
                errors.append("scheme_b_quant_kv_attention.quantize_prob_bits must be 8")
            if bool(kv_attention.get("fallback_to_fp_attention", True)):
                errors.append("scheme_b_quant_kv_attention.fallback_to_fp_attention must be False")
            if bool(kv_attention.get("fallback_to_scale_delayed_v", True)):
                errors.append("scheme_b_quant_kv_attention.fallback_to_scale_delayed_v must be False")
        if not quant_mode and bool(owner.scheme_b_quant_kv_attention_enabled):
            errors.append(f"{self.profile_mode} requires scheme_b_quant_kv_attention.enabled=False")
        base_model = owner.model.icae.get_base_model().model
        suffix_layers = list(range(base_model.gnn_start_layer, base_model.config.num_hidden_layers))
        if suffix_layers != list(range(26, 32)):
            errors.append(f"suffix Transformer layers must be 26-31, got {suffix_layers}")
        if bool(getattr(owner, "gofa_query_trace_enabled", False)):
            errors.append("gofa_query_trace.enabled must be False to keep canonical traces read-only")
        if self.config.get("export_gpu_time", True) and not torch.cuda.is_available():
            errors.append("export_gpu_time=True requires CUDA")
        if self.config.get("export_detail_gpu_time", True) and not self.config.get("export_gpu_time", True):
            errors.append("export_detail_gpu_time=True requires export_gpu_time=True")
        if errors:
            raise ValueError("Invalid canonical H100 per-query latency configuration: " + "; ".join(errors) + ".")
        self._initialize_csv()
        self.csv_initialized = True

    def set_context(self, task=None, split=None, query_index=None, is_warmup=False):
        split = normalize_trace_split(split)
        key = (task, split)
        query_index = int(query_index) if query_index is not None else None
        if is_warmup:
            self.context = {
                "task": task,
                "split": split,
                "query_index": query_index,
                "rep": None,
                "is_warmup": True,
            }
            return
        if key not in self.rep_by_split:
            self.rep_by_split[key] = int(self.existing_next_rep.get(key, 0))
        elif query_index == 0 and self.last_query_index.get(key) is not None:
            self.rep_by_split[key] += 1
        if query_index is not None:
            self.last_query_index[key] = query_index
        self.context = {
            "task": task,
            "split": split,
            "query_index": query_index,
            "rep": self.rep_by_split[key],
            "is_warmup": False,
        }

    def _catalog(self, task):
        if task not in self.catalogs:
            path = resolve_task_path(self.config["trace_index_path"], task)
            self.catalogs[task] = load_trace_index_catalog(
                path,
                expected_task=task,
                strict=bool(self.config.get("strict_trace_match", True)),
            )
        return self.catalogs[task]

    def _sync(self, device):
        if (
            self.config.get("cuda_sync", True)
            and device is not None
            and device.type == "cuda"
        ):
            torch.cuda.synchronize(device)

    def begin_query(self, graph, device):
        if not self.enabled or self.context.get("is_warmup", False):
            return False
        if self.current is not None:
            raise RuntimeError("GOFA per-query latency exporter received overlapping queries.")
        task = self.context.get("task")
        split = self.context.get("split")
        query_index = self.context.get("query_index")
        if split not in {"val", "test"}:
            return False
        if infer_graph_batch_size(graph) != 1:
            raise RuntimeError(FORMAL_BATCH_ERROR)
        if not task or query_index is None:
            raise RuntimeError("GOFA per-query latency exporter requires task, split, and query_index context.")

        catalog = self._catalog(task)
        expected = catalog["by_query"].get((split, query_index))
        if expected is None:
            raise RuntimeError(
                "GOFA per-query latency trace match failed: "
                f"task={task}, split={split}, query_index={query_index} is absent from {catalog['path']}"
            )
        runtime_graph_signature = graph_signature(task, split, query_index, graph=graph)
        runtime_query_uid = query_uid(task, split, query_index, runtime_graph_signature)
        if (
            self.config.get("strict_trace_match", True)
            and runtime_graph_signature != expected["graph_signature"]
        ):
            raise RuntimeError(
                "GOFA per-query latency graph signature mismatch: "
                f"task={task}, split={split}, query_index={query_index}, "
                f"runtime={runtime_graph_signature}, trace={expected['graph_signature']}"
            )
        if (
            self.config.get("strict_trace_match", True)
            and runtime_query_uid != expected["query_uid"]
        ):
            raise RuntimeError(
                "GOFA per-query latency query UID mismatch: "
                f"runtime={runtime_query_uid}, trace={expected['query_uid']}"
            )
        runtime_profile = normalize_workload_profile(
            getattr(self.owner.model_args, "workload_profile", None)
        )["name"]
        if (
            self.config.get("strict_trace_match", True)
            and expected["workload_profile"]
            and runtime_profile != expected["workload_profile"]
        ):
            raise RuntimeError(
                "GOFA per-query latency workload profile mismatch: "
                f"runtime={runtime_profile}, trace={expected['workload_profile']}"
            )
        rep = int(self.context.get("rep", 0))
        previous_order = self.last_trace_order.get((task, rep), -1)
        if self.config.get("strict_trace_match", True) and expected["trace_order"] != previous_order + 1:
            raise RuntimeError(
                "GOFA per-query latency trace order mismatch: "
                f"task={task}, rep={rep}, expected_next={previous_order + 1}, "
                f"actual={expected['trace_order']}. Validation must run before test."
            )

        self._sync(device)
        query_start_ns = time.perf_counter_ns()
        query_gpu_start = None
        query_gpu_end = None
        if self.config.get("export_gpu_time", True):
            if device is None or device.type != "cuda":
                raise RuntimeError("GOFA per-query GPU latency export requires a CUDA model device.")
            query_gpu_start = torch.cuda.Event(enable_timing=True)
            query_gpu_end = torch.cuda.Event(enable_timing=True)
            query_gpu_start.record(torch.cuda.current_stream(device))

        base_model = self.owner.model.icae.get_base_model().model
        base_model.begin_per_query_latency_capture(
            export_gpu_time=bool(self.config.get("export_gpu_time", True)),
            export_detail_gpu_time=bool(self.config.get("export_detail_gpu_time", True)),
        )
        stats = getattr(base_model, "quant_kv_attention_stats", {})
        int_gemm_stats = getattr(getattr(self.owner, "scheme_b_int_gemm_quantizer", None), "stats", {})
        self.current = {
            "device": device,
            "expected": expected,
            "graph_signature": runtime_graph_signature,
            "workload_profile": runtime_profile,
            "rep": rep,
            "query_start_ns": query_start_ns,
            "query_gpu_start": query_gpu_start,
            "query_gpu_end": query_gpu_end,
            "encoder_wall_ms": 0.0,
            "decoder_wall_ms": 0.0,
            "cache": None,
            "quant_calls_start": int(stats.get("quant_kv_attention_call_count", 0)),
            "fallback_start": int(stats.get("fallback_count", 0)) + int(stats.get("fallback_count_pv", 0)),
            "int_gemm_fallback_start": int(int_gemm_stats.get("fallback_count", 0)),
            "int_gemm_calls_start": int(int_gemm_stats.get("int_gemm_call_count", 0)),
        }
        return True

    def active(self):
        return self.current is not None

    def stage_start(self, device):
        if not self.active():
            return None
        self._sync(device)
        return time.perf_counter_ns()

    def stage_end(self, name, start_ns, device):
        if not self.active() or start_ns is None:
            return
        self._sync(device)
        self.current[f"{name}_wall_ms"] = (time.perf_counter_ns() - start_ns) / 1.0e6

    def record_cache_snapshot(
        self,
        timing,
        cache_hits,
        cache_misses,
        cache_skips,
        cache_keys,
        cache_fallback_count=0,
    ):
        if not self.active():
            return
        cache_keys = list(cache_keys or [])
        self.current["cache"] = {
            "cache_load_wall_ms": 1000.0 * (
                float(timing.get("load_s", 0.0))
                + float(timing.get("delta_load_s", 0.0))
                + float(timing.get("dequant_s", 0.0))
            ),
            "online_nog_prefix_wall_ms": 1000.0 * float(timing.get("miss_compute_s", 0.0)),
            "cache_assembly_wall_ms": 1000.0 * float(timing.get("assemble_s", 0.0)),
            "suffix_wall_ms": 1000.0 * float(timing.get("suffix_compute_s", 0.0)),
            "cache_hits": int(cache_hits),
            "cache_misses": int(cache_misses),
            "cache_skips": int(cache_skips),
            "cache_fallback_count": int(cache_fallback_count),
            "cache_keys": cache_keys,
        }

    def _elapsed_event_pairs(self, pairs):
        return elapsed_event_pairs_ms(pairs)

    def finish_query(self):
        if not self.active():
            return None
        current = self.current
        device = current["device"]
        if current["query_gpu_end"] is not None:
            current["query_gpu_end"].record(torch.cuda.current_stream(device))
        self._sync(device)
        query_wall_ms = (time.perf_counter_ns() - current["query_start_ns"]) / 1.0e6
        if current["query_gpu_end"] is not None and not self.config.get("cuda_sync", True):
            current["query_gpu_end"].synchronize()

        base_model = self.owner.model.icae.get_base_model().model
        suffix_events = base_model.end_per_query_latency_capture()
        query_gpu_ms = 0.0
        suffix_gnn_gpu_ms = 0.0
        suffix_transformer_gpu_ms = 0.0
        detail_gpu_ms = {
            "prefix_transformer_gpu_ms": 0.0,
            "dense_fc_gpu_ms": 0.0,
            "attention_gpu_ms": 0.0,
            "norm_residual_other_gpu_ms": 0.0,
            "quant_kv_attention_gpu_ms": 0.0,
            "kv_prepare_gpu_ms": 0.0,
            "int_qk_gpu_ms": 0.0,
            "softmax_prob_quant_gpu_ms": 0.0,
            "int_pv_gpu_ms": 0.0,
            "gnn_score_gpu_ms": 0.0,
            "gnn_message_gpu_ms": 0.0,
            "gnn_update_gpu_ms": 0.0,
            "gnn_other_gpu_ms": 0.0,
        }
        if current["query_gpu_start"] is not None:
            query_gpu_ms = float(current["query_gpu_start"].elapsed_time(current["query_gpu_end"]))
            suffix_gnn_gpu_ms = self._elapsed_event_pairs(suffix_events.get("gnn", []))
            suffix_transformer_gpu_ms = self._elapsed_event_pairs(suffix_events.get("transformer", []))
            if self.config.get("export_detail_gpu_time", True):
                category_to_field = {
                    "prefix_transformer": "prefix_transformer_gpu_ms",
                    "dense_fc": "dense_fc_gpu_ms",
                    "attention": "attention_gpu_ms",
                    "quant_kv_attention": "quant_kv_attention_gpu_ms",
                    "kv_prepare": "kv_prepare_gpu_ms",
                    "int_qk": "int_qk_gpu_ms",
                    "softmax_prob_quant": "softmax_prob_quant_gpu_ms",
                    "int_pv": "int_pv_gpu_ms",
                    "gnn_score": "gnn_score_gpu_ms",
                    "gnn_message": "gnn_message_gpu_ms",
                    "gnn_update": "gnn_update_gpu_ms",
                }
                for category, field_name in category_to_field.items():
                    detail_gpu_ms[field_name] = self._elapsed_event_pairs(suffix_events.get(category, []))
                classified_gnn_ms = (
                    detail_gpu_ms["gnn_score_gpu_ms"]
                    + detail_gpu_ms["gnn_message_gpu_ms"]
                    + detail_gpu_ms["gnn_update_gpu_ms"]
                )
                detail_gpu_ms["gnn_other_gpu_ms"] = max(0.0, suffix_gnn_gpu_ms - classified_gnn_ms)
                transformer_total_ms = (
                    detail_gpu_ms["prefix_transformer_gpu_ms"] + suffix_transformer_gpu_ms
                )
                detail_gpu_ms["norm_residual_other_gpu_ms"] = max(
                    0.0,
                    transformer_total_ms
                    - detail_gpu_ms["attention_gpu_ms"]
                    - detail_gpu_ms["dense_fc_gpu_ms"],
                )

        cache = current.get("cache")
        if cache is None:
            if self.profile_mode in CACHE_PROFILE_MODES:
                self.current = None
                raise RuntimeError("GOFA cache profile did not receive a Scheme-B cache snapshot.")
            cache = {
                "cache_load_wall_ms": 0.0,
                "online_nog_prefix_wall_ms": 0.0,
                "cache_assembly_wall_ms": 0.0,
                "suffix_wall_ms": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_skips": 0,
                "cache_fallback_count": 0,
                "cache_keys": [],
            }
        runtime_keys = cache["cache_keys"]
        expected_keys = current["expected"]["cache_keys"]
        if (
            self.profile_mode in CACHE_PROFILE_MODES
            and self.config.get("strict_trace_match", True)
            and runtime_keys != expected_keys
        ):
            first_mismatch = None
            for index in range(max(len(runtime_keys), len(expected_keys))):
                runtime_key = runtime_keys[index] if index < len(runtime_keys) else None
                expected_key = expected_keys[index] if index < len(expected_keys) else None
                if runtime_key != expected_key:
                    first_mismatch = {
                        "item_index": index,
                        "runtime_cache_key": runtime_key,
                        "trace_cache_key": expected_key,
                    }
                    break
            self.current = None
            raise RuntimeError(
                "GOFA per-query latency cache-key trace mismatch: "
                f"task={self.context.get('task')}, split={self.context.get('split')}, "
                f"query_index={self.context.get('query_index')}, runtime_items={len(runtime_keys)}, "
                f"trace_items={len(expected_keys)}, first_mismatch={first_mismatch}"
            )
        stats = getattr(base_model, "quant_kv_attention_stats", {})
        int_gemm_stats = getattr(getattr(self.owner, "scheme_b_int_gemm_quantizer", None), "stats", {})
        quant_calls = int(stats.get("quant_kv_attention_call_count", 0)) - current["quant_calls_start"]
        fallback_count = (
            int(stats.get("fallback_count", 0))
            + int(stats.get("fallback_count_pv", 0))
            - current["fallback_start"]
            + int(int_gemm_stats.get("fallback_count", 0))
            - current["int_gemm_fallback_start"]
            + int(cache.get("cache_fallback_count", 0))
        )
        int_gemm_call_count = (
            int(int_gemm_stats.get("int_gemm_call_count", 0)) - current["int_gemm_calls_start"]
        )
        if self.profile_mode in CACHE_PROFILE_MODES and int(cache.get("cache_misses", 0)) != 0:
            self.current = None
            raise RuntimeError(
                f"GOFA {self.profile_mode} observed cache_misses={cache.get('cache_misses')}"
            )
        if self.profile_mode in QUANT_PROFILE_MODES and fallback_count != 0:
            self.current = None
            raise RuntimeError(
                "GOFA canonical H100 per-query latency observed quantized-KV fallback: "
                f"fallback_count={fallback_count}"
            )

        expected = current["expected"]
        row = {
            "task": expected["task"],
            "split": expected["split"],
            "trace_order": expected["trace_order"],
            "query_index": expected["query_index"],
            "query_uid": expected["query_uid"],
            "rep": current["rep"],
            "profile_mode": self.profile_mode,
            "workload_profile": current["workload_profile"],
            "graph_signature": current["graph_signature"],
            "query_wall_ms": query_wall_ms if self.config.get("export_wall_time", True) else 0.0,
            "query_gpu_ms": query_gpu_ms if self.config.get("export_gpu_time", True) else 0.0,
            "encoder_wall_ms": current["encoder_wall_ms"] if self.config.get("export_wall_time", True) else 0.0,
            "decoder_wall_ms": current["decoder_wall_ms"] if self.config.get("export_wall_time", True) else 0.0,
            "suffix_gnn_gpu_ms": suffix_gnn_gpu_ms if self.config.get("export_gpu_time", True) else 0.0,
            "suffix_transformer_gpu_ms": (
                suffix_transformer_gpu_ms if self.config.get("export_gpu_time", True) else 0.0
            ),
            "quant_kv_attention_calls": quant_calls,
            "fallback_count": fallback_count,
            "int_gemm_call_count": int_gemm_call_count,
            "logical_memory_loaded_bytes": (
                expected["logical_memory_loaded_bytes"] if self.profile_mode in QUANT_PROFILE_MODES else 0
            ),
            "logical_key_loaded_bytes": (
                expected["logical_key_loaded_bytes"] if self.profile_mode in QUANT_PROFILE_MODES else 0
            ),
            "logical_value_loaded_bytes": (
                expected["logical_value_loaded_bytes"] if self.profile_mode in QUANT_PROFILE_MODES else 0
            ),
        }
        row.update({
            key: value
            for key, value in cache.items()
            if key not in {"cache_keys", "cache_fallback_count"}
        })
        row.update(detail_gpu_ms)
        with open(self.config["output_csv"], "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writerow({key: row[key] for key in CSV_FIELDS})
            handle.flush()
        self.last_trace_order[(expected["task"], current["rep"])] = expected["trace_order"]
        self.current = None
        print(
            "GOFA per-query latency written: "
            f"task={row['task']}, split={row['split']}, trace_order={row['trace_order']}, "
            f"query_index={row['query_index']}, rep={row['rep']}, "
            f"query_wall_ms={row['query_wall_ms']:.3f}, query_gpu_ms={row['query_gpu_ms']:.3f}, "
            f"path={self.config['output_csv']}"
        )
        return row

    def abort_query(self):
        if self.current is None:
            return
        base_model = self.owner.model.icae.get_base_model().model
        base_model.abort_per_query_latency_capture()
        self.current = None
