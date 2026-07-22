import argparse
import os
from datetime import timedelta

from lightning.pytorch.loggers import WandbLogger

from gp.utils.utils import (load_yaml, combine_dict, merge_mod, setup_exp, set_random_seed, )
from gp.lightning.metric import (EvalKit, )
from gp.lightning.data_template import DataModule
from gp.lightning.training import lightning_fit, lightning_test
from gp.lightning.module_template import ExpConfig
from lightning_model import GraphTextPredLightning
from model import GOFA
from modules.gofa import GOFAMistralConfig
from modules.gofa.workload_profile import (
    resolve_from_saved_flags,
    resolve_saved_workload_names,
    validate_runtime_sampling,
    workload_profile_from_runtime,
)

from torchmetrics import AUROC, Accuracy, MeanMetric, MeanAbsoluteError, Perplexity
from utils import (MultiApr, MultiAuc, SimAnyAuc, normalized_loss_factory, sentence_base, sentence_perplexity)
from gp.lightning.data_template import DataWithMeta
from tasks import GOFAPretrainTaskWrapper, GOFAFineTuneTaskWrapper
from TAGLAS import get_evaluators
from TAGLAS.data import TAGData
import torch
from types import SimpleNamespace


def _first_config_value(value, field_name):
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"Configuration field {field_name!r} must not be empty")
        return value[0]
    return value


def _resolve_train_sampling_config(params):
    """Resolve train sampling fields, using inference equivalents when absent."""
    fallbacks = {
        "hops": ("inf_hops",),
        "train_max_nodes_per_hops": ("max_nodes_per_hop", "inf_max_nodes_per_hops"),
        "instructs": ("inf_instructs",),
        "selections": ("inf_selections",),
    }
    resolved = {}
    for field_name, fallback_names in fallbacks.items():
        value = getattr(params, field_name, None)
        source_name = field_name
        if value is None:
            for fallback_name in fallback_names:
                fallback_value = getattr(params, fallback_name, None)
                if fallback_value is not None:
                    value = fallback_value
                    source_name = fallback_name
                    break
        if value is None:
            fallback_text = ", ".join(repr(name) for name in fallback_names)
            raise AttributeError(
                f"Missing configuration field {field_name!r}; expected it directly or via {fallback_text}"
            )
        resolved[field_name] = _first_config_value(value, source_name)
    return resolved


def _gofa_per_query_latency_enabled(params):
    nested = getattr(params, "gofa_per_query_latency", None)
    enabled = nested.get("enabled", False) if isinstance(nested, dict) else False
    flat = getattr(params, "gofa_per_query_latency_enabled", None)
    return bool(enabled if flat is None else flat)


def _gofa_query_trace_enabled(params):
    nested = getattr(params, "gofa_query_trace", None)
    enabled = nested.get("enabled", False) if isinstance(nested, dict) else False
    flat = getattr(params, "gofa_query_trace_enabled", None)
    return bool(enabled if flat is None else flat)


def _validate_canonical_h100_latency_run(params):
    if not (_gofa_per_query_latency_enabled(params) or _gofa_query_trace_enabled(params)):
        return
    errors = []
    if getattr(params, "run_mode", None) != "inf":
        errors.append("run_mode must be inf")
    if int(getattr(params, "batch_size", -1)) != 1:
        errors.append("batch_size must be 1")
    if bool(getattr(params, "skip_validation", False)):
        errors.append("skip_validation must be False so validation runs before test")
    canonical_tasks = {"cora_node", "cora_link", "pubmed_node", "wikics", "arxiv"}
    tasks = list(getattr(params, "eval_task_names", []) or [])
    if not tasks:
        errors.append("eval_task_names must contain at least one canonical task")
    unexpected_tasks = sorted(set(tasks) - canonical_tasks)
    if unexpected_tasks:
        errors.append(f"unsupported canonical task(s): {unexpected_tasks}")

    profile = workload_profile_from_runtime(params)
    errors.extend(validate_runtime_sampling(
        profile,
        seed=getattr(params, "seed", -1),
        eval_sample_size=getattr(params, "eval_sample_size", -1),
        tasks=tasks,
        sample_sizes=getattr(params, "inf_sample_size_per_task", None),
        hops=getattr(params, "inf_hops", None),
        max_nodes=getattr(params, "inf_max_nodes_per_hops", None),
    ))
    if errors:
        raise ValueError("Invalid canonical H100 per-query latency run: " + "; ".join(errors) + ".")


def main(params):
    if getattr(params, "model_type", None) == "llm_n" or getattr(params, "llm_n_enabled", False):
        # Isolated pure-LLM path.  The default GOFA construction below remains
        # untouched and is never instantiated for LLM-N runs.
        from llm_n.runner import run as run_llm_n
        return run_llm_n(params)

    ##################################################################
    #                    Configuration                               #
    ##################################################################

    if params.base_llm == 'mistral7b':
        from modules.gofa.gofa import TrainingArguments
        from modules.gofa.gofa import ModelArguments
        gofa_config = GOFAMistralConfig
    else:
        raise NotImplementedError(params.base_llm + " is not supported. Please choose from: mistral7b,")
    if params.mode == "generate":
        params.last_save = False
    params.workload_profile = workload_profile_from_runtime(params)
    _validate_canonical_h100_latency_run(params)

    wandb_logger = WandbLogger(project=params.log_project, name=f"{params.exp_name}_{params.base_llm}",
                               save_dir=params.exp_dir, offline=params.offline_log, )
    print("available devices: ", torch.cuda.device_count())
    if params.ckpt_save_path is not None:
        date = params.exp_dir.split("/")[-1]
        params.ckpt_save_path = params.ckpt_save_path + "/" + date
    params_dict = vars(params)
    wandb_logger.log_table(key="hparams", columns=list(params_dict.keys()), data=[list(params_dict.values())])
    model_args, training_args, gofa_args = ModelArguments(), TrainingArguments(), gofa_config(
        num_layers=params.num_layers, gnn_type=params.gnn_type, fuse_type=params.fuse_type)
    model_args.workload_profile = dict(params.workload_profile)
    model_args.dec_lora = params.dec_lora
    if getattr(params, "model_name_or_path", None):
        model_args.model_name_or_path = params.model_name_or_path
    if getattr(params, "attn_implementation", None):
        model_args.attn_implementation = params.attn_implementation
    if getattr(params, "checkpoint_dir", None):
        model_args.checkpoint_dir = params.checkpoint_dir
    if hasattr(params, "use_encoder_cache"):
        model_args.use_encoder_cache = params.use_encoder_cache
    if getattr(params, "encoder_cache_dir", None):
        model_args.encoder_cache_dir = params.encoder_cache_dir
    if getattr(params, "encoder_cache_tag", None):
        model_args.encoder_cache_tag = params.encoder_cache_tag
    if hasattr(params, "encoder_cache_skip_nog"):
        model_args.encoder_cache_skip_nog = params.encoder_cache_skip_nog
    if getattr(params, "encoder_cache_mode", None):
        model_args.encoder_cache_mode = params.encoder_cache_mode
    if hasattr(params, "encoder_cache_manifest"):
        model_args.encoder_cache_manifest = params.encoder_cache_manifest
    for field_name in (
        "encoder_cache_manifest_enabled",
        "encoder_cache_manifest_output_path",
        "encoder_cache_manifest_append",
        "encoder_cache_manifest_log_interval",
    ):
        if hasattr(params, field_name):
            setattr(model_args, field_name, getattr(params, field_name))
    if hasattr(params, "gofa_trace_source_audit"):
        model_args.gofa_trace_source_audit = params.gofa_trace_source_audit
    for field_name in (
        "gofa_trace_source_audit_enabled",
        "gofa_trace_source_audit_output_dir",
        "gofa_trace_source_audit_max_queries",
        "gofa_trace_source_audit_include_full_arrays",
        "gofa_trace_source_audit_flush_interval",
        "gofa_trace_source_audit_rank_zero_only",
        "gofa_trace_source_audit_strict",
        "gofa_trace_source_audit_dump_pre_cache_snapshot",
        "gofa_trace_source_audit_dump_cache_miss_snapshot",
    ):
        if hasattr(params, field_name):
            setattr(model_args, field_name, getattr(params, field_name))
    if hasattr(params, "gofa_query_trace"):
        model_args.gofa_query_trace = params.gofa_query_trace
    for field_name in (
        "gofa_query_trace_enabled",
        "gofa_query_trace_output_dir",
        "gofa_query_trace_max_queries",
        "gofa_query_trace_include_token_ids",
        "gofa_query_trace_include_text_preview",
        "gofa_query_trace_rank_zero_only",
        "gofa_query_trace_strict",
    ):
        if hasattr(params, field_name):
            setattr(model_args, field_name, getattr(params, field_name))
    if hasattr(params, "gofa_per_query_latency"):
        model_args.gofa_per_query_latency = params.gofa_per_query_latency
    for field_name in (
        "gofa_per_query_latency_enabled",
        "gofa_per_query_latency_profile_mode",
        "gofa_per_query_latency_output_csv",
        "gofa_per_query_latency_trace_index_path",
        "gofa_per_query_latency_strict_trace_match",
        "gofa_per_query_latency_cuda_sync",
        "gofa_per_query_latency_export_wall_time",
        "gofa_per_query_latency_export_gpu_time",
        "gofa_per_query_latency_export_detail_gpu_time",
        "gofa_per_query_latency_append",
        "gofa_per_query_latency_rank_zero_only",
    ):
        if hasattr(params, field_name):
            setattr(model_args, field_name, getattr(params, field_name))
    for field_name in (
        "eval_task_names",
        "train_task_names",
        "run_mode",
    ):
        if hasattr(params, field_name):
            setattr(model_args, field_name, getattr(params, field_name))
    if hasattr(params, "encoder_cache_verify"):
        model_args.encoder_cache_verify = params.encoder_cache_verify
    if hasattr(params, "encoder_cache_verify_tolerance"):
        model_args.encoder_cache_verify_tolerance = params.encoder_cache_verify_tolerance
    if hasattr(params, "encoder_cache_verify_mean_tolerance"):
        model_args.encoder_cache_verify_mean_tolerance = params.encoder_cache_verify_mean_tolerance
    if hasattr(params, "encoder_cache_verify_p99_tolerance"):
        model_args.encoder_cache_verify_p99_tolerance = params.encoder_cache_verify_p99_tolerance
    if hasattr(params, "encoder_cache_verify_relative_l2_tolerance"):
        model_args.encoder_cache_verify_relative_l2_tolerance = params.encoder_cache_verify_relative_l2_tolerance
    if hasattr(params, "encoder_cache_verify_max_tolerance"):
        model_args.encoder_cache_verify_max_tolerance = params.encoder_cache_verify_max_tolerance
    if hasattr(params, "encoder_cache_verify_log_interval"):
        model_args.encoder_cache_verify_log_interval = params.encoder_cache_verify_log_interval
    if hasattr(params, "encoder_cache_verify_quantile_sample_size"):
        model_args.encoder_cache_verify_quantile_sample_size = params.encoder_cache_verify_quantile_sample_size
    if hasattr(params, "profile_stage_times"):
        model_args.profile_stage_times = params.profile_stage_times
    if hasattr(params, "profile_stage_log_interval"):
        model_args.profile_stage_log_interval = params.profile_stage_log_interval
    if hasattr(params, "profile_memory_kv_transformer_breakdown"):
        model_args.profile_memory_kv_transformer_breakdown = params.profile_memory_kv_transformer_breakdown
    if hasattr(params, "scheme_b_quant"):
        model_args.scheme_b_quant = params.scheme_b_quant
    for field_name in (
        "scheme_b_quant_enabled",
        "scheme_b_quant_base_bits",
        "scheme_b_quant_delta_bits",
        "scheme_b_quant_memory_base_bits",
        "scheme_b_quant_key_base_bits",
        "scheme_b_quant_value_base_bits",
        "scheme_b_quant_memory_delta_bits",
        "scheme_b_quant_key_delta_bits",
        "scheme_b_quant_value_delta_bits",
        "scheme_b_quant_static_high_ratio",
        "scheme_b_quant_static_mid_ratio",
        "scheme_b_quant_target_aware_delta",
        "scheme_b_quant_target_aware_policy",
        "scheme_b_quant_target_delta_hops",
        "scheme_b_quant_keep_target_edges",
        "scheme_b_quant_local_degree_top_ratio",
        "scheme_b_quant_local_degree_threshold",
        "scheme_b_quant_max_delta_items_per_batch",
        "scheme_b_quant_cache_dir",
        "scheme_b_quant_fake_quant",
        "scheme_b_quant_debug_zero_base",
        "scheme_b_quant_strict",
        "scheme_b_quant_load_key_base",
        "scheme_b_quant_load_value_base",
        "scheme_b_quant_kv_base_load_policy",
        "scheme_b_quant_kv_base_target_hops",
        "scheme_b_quant_kv_base_local_degree_top_ratio",
        "scheme_b_quant_kv_base_local_degree_threshold",
        "scheme_b_quant_kv_base_keep_target_edges",
        "scheme_b_quant_load_memory_delta",
        "scheme_b_quant_load_key_delta",
        "scheme_b_quant_load_value_delta",
    ):
        if hasattr(params, field_name):
            setattr(model_args, field_name, getattr(params, field_name))
    if hasattr(params, "scheme_b_weight_quant"):
        model_args.scheme_b_weight_quant = params.scheme_b_weight_quant
    for field_name in (
        "scheme_b_weight_quant_enabled",
        "scheme_b_weight_quant_bits",
        "scheme_b_weight_quant_target",
        "scheme_b_weight_quant_fake_quant",
        "scheme_b_weight_quant_quantize_attention",
        "scheme_b_weight_quant_quantize_mlp",
        "scheme_b_weight_quant_quantize_layernorm",
        "scheme_b_weight_quant_log_quantized_modules",
    ):
        if hasattr(params, field_name):
            setattr(model_args, field_name, getattr(params, field_name))
    if hasattr(params, "scheme_b_activation_quant"):
        model_args.scheme_b_activation_quant = params.scheme_b_activation_quant
    for field_name in (
        "scheme_b_activation_quant_enabled",
        "scheme_b_activation_quant_bits",
        "scheme_b_activation_quant_target",
        "scheme_b_activation_quant_fake_quant",
        "scheme_b_activation_quant_quantize_attention",
        "scheme_b_activation_quant_quantize_q_proj",
        "scheme_b_activation_quant_quantize_k_proj",
        "scheme_b_activation_quant_quantize_v_proj",
        "scheme_b_activation_quant_quantize_o_proj",
        "scheme_b_activation_quant_quantize_mlp",
        "scheme_b_activation_quant_quantize_qkv_outputs",
        "scheme_b_activation_quant_quantize_attn_output",
        "scheme_b_activation_quant_quantize_mlp_output",
        "scheme_b_activation_quant_per_token",
        "scheme_b_activation_quant_clip_ratio",
        "scheme_b_activation_quant_q_proj_bits",
        "scheme_b_activation_quant_k_proj_bits",
        "scheme_b_activation_quant_v_proj_bits",
        "scheme_b_activation_quant_o_proj_bits",
        "scheme_b_activation_quant_mlp_bits",
        "scheme_b_activation_quant_log_quantized_modules",
    ):
        if hasattr(params, field_name):
            setattr(model_args, field_name, getattr(params, field_name))
    if hasattr(params, "scheme_b_int_gemm"):
        model_args.scheme_b_int_gemm = params.scheme_b_int_gemm
    for field_name in (
        "scheme_b_int_gemm_enabled",
        "scheme_b_int_gemm_target",
        "scheme_b_int_gemm_weight_bits",
        "scheme_b_int_gemm_activation_bits",
        "scheme_b_int_gemm_backend",
        "scheme_b_int_gemm_quantize_attention",
        "scheme_b_int_gemm_quantize_mlp",
        "scheme_b_int_gemm_quantize_layernorm",
        "scheme_b_int_gemm_fallback_to_fake_quant",
        "scheme_b_int_gemm_log_modules",
        "scheme_b_int_gemm_log_interval",
    ):
        if hasattr(params, field_name):
            setattr(model_args, field_name, getattr(params, field_name))
    if hasattr(params, "scheme_b_quant_kv_attention"):
        model_args.scheme_b_quant_kv_attention = params.scheme_b_quant_kv_attention
    for field_name in (
        "scheme_b_quant_kv_attention_enabled",
        "scheme_b_quant_kv_attention_backend",
        "scheme_b_quant_kv_attention_key_scale_fold_into_q",
        "scheme_b_quant_kv_attention_quantize_query_bits",
        "scheme_b_quant_kv_attention_key_bits",
        "scheme_b_quant_kv_attention_value_bits",
        "scheme_b_quant_kv_attention_use_int_qk",
        "scheme_b_quant_kv_attention_pv_compute_mode",
        "scheme_b_quant_kv_attention_quantize_prob_bits",
        "scheme_b_quant_kv_attention_prob_quant_granularity",
        "scheme_b_quant_kv_attention_prob_quant_unsigned",
        "scheme_b_quant_kv_attention_prob_quant_qmax",
        "scheme_b_quant_kv_attention_fallback_to_fp_attention",
        "scheme_b_quant_kv_attention_fallback_to_scale_delayed_v",
        "scheme_b_quant_kv_attention_compare_int_pv_with_fp_pv",
        "scheme_b_quant_kv_attention_log_interval",
    ):
        if hasattr(params, field_name):
            setattr(model_args, field_name, getattr(params, field_name))
    if hasattr(params, "scheme_b_activation_observer"):
        model_args.scheme_b_activation_observer = params.scheme_b_activation_observer
    for field_name in (
        "scheme_b_activation_observer_enabled",
        "scheme_b_activation_observer_output_dir",
        "scheme_b_activation_observer_max_batches",
        "scheme_b_activation_observer_max_items_per_module",
        "scheme_b_activation_observer_target",
        "scheme_b_activation_observer_layers",
        "scheme_b_activation_observer_projections",
        "scheme_b_activation_observer_save_tensor",
        "scheme_b_activation_observer_save_stats",
        "scheme_b_activation_observer_sample_tokens",
        "scheme_b_activation_observer_sample_channels",
        "scheme_b_activation_observer_compute_quant_error",
        "scheme_b_activation_observer_quant_bits",
        "scheme_b_activation_observer_per_token",
        "scheme_b_activation_observer_clip_ratio",
        "scheme_b_activation_observer_log_interval",
    ):
        if hasattr(params, field_name):
            setattr(model_args, field_name, getattr(params, field_name))
    if hasattr(params, "scheme_b_ablation"):
        model_args.scheme_b_ablation = params.scheme_b_ablation
    for field_name in (
        "scheme_b_ablation_enabled",
        "scheme_b_ablation_mode",
        "scheme_b_ablation_zero_memory_state",
        "scheme_b_ablation_zero_text_kv",
        "scheme_b_ablation_zero_edge_cache",
        "scheme_b_ablation_keep_target_edges",
        "scheme_b_ablation_log_interval",
    ):
        if hasattr(params, field_name):
            setattr(model_args, field_name, getattr(params, field_name))
    training_args.model_max_length = params.llm_max_length
    if params.training_precision == "bf16-mixed":
        training_args.bf16 = True
        gofa_args.llama_dtype = torch.bfloat16
    else:
        training_args.bf16 = False
        gofa_args.llama_dtype = torch.float16

    ##################################################################
    #                    Create datasets                             #
    ##################################################################
    def data_size_filter(data: TAGData, **kwargs):
        estimated_mem = 24.495 + 0.4645 * len(data.node_map) + 0.0042 * len(torch.unique(data.node_map)) + 0.1689 * len(
            data.edge_map) + 0.2846 * len(torch.unique(data.edge_map))
        if len(data.node_map) + len(torch.unique(data.edge_map)) < 40 and estimated_mem < 65:
            return data
        else:
            return None

    if params.run_mode == "pretrain":
        ######################################################################################################
        #                                          Pretrain Task                                             #
        ######################################################################################################
        task_names = ["mag240m", "mag240m", "mag240m", "arxiv", "arxiv", "arxiv", "pubmed_node", "pubmed_node",
                      "pubmed_node", "wiki_graph", "wiki_graph", "wiki_graph", "wikikg90m", "wikikg90m", "wikikg90m",
                      "ultrachat200k"]

        save_names = ["pretrain_", "pretrain_IR_kc_", "pretrain_IR_ck_", "pretrain_", "pretrain_IR_kc_",
                      "pretrain_IR_ck_", "pretrain_", "pretrain_IR_kc_", "pretrain_IR_ck_", "pretrain_",
                      "pretrain_IR_kc_", "pretrain_IR_ck_", "pretrain_", "pretrain_IR_kc_", "pretrain_IR_ck_",
                      "pretrain_"]

        filter_func = data_size_filter
        save_names = [name + str(params.last_epochs) for name in save_names]
        train_task = GOFAPretrainTaskWrapper(task_names, root=params.data_root_path, save_name=save_names,
                                             fast_data_load=True, filter_func=filter_func)

        val_tasks = GOFAPretrainTaskWrapper("cora", root=params.data_root_path, split="val", sample_size=100,
                                            save_name="pretrain_val", pretrain_tasks=["CS", "CN", "SP"],
                                            num_workers=params.num_workers, num_additional_sentences=3, num_SP=3,
                                            num_CN=3)

        test_tasks = GOFAPretrainTaskWrapper("cora", root=params.data_root_path, split="test", sample_size=100,
                                             save_name="pretrain_test", pretrain_tasks=["CS", "CN", "SP"],
                                             num_workers=params.num_workers, num_additional_sentences=3, num_SP=3,
                                             num_CN=3)

        n_steps = int(len(train_task) * params.num_epochs / (params.grad_acc_step * int(torch.cuda.device_count())))

        train_task = DataWithMeta(train_task, batch_size=params.batch_size, sample_size=params.train_sample_size)

        val_tasks = [
            DataWithMeta(val_tasks, batch_size=params.batch_size, sample_size=params.eval_sample_size, state_name="val",
                         metric="perp", classes=32132, meta_data={"eval_func": sentence_perplexity})]

        test_tasks = [DataWithMeta(test_tasks, batch_size=params.batch_size, sample_size=params.eval_sample_size,
                                   state_name="test", metric="perp", classes=32132,
                                   meta_data={"eval_func": sentence_perplexity})]
        evlter = []


    else:
        train_tasks = params.train_task_names
        eval_tasks = params.eval_task_names
        train_sampling = _resolve_train_sampling_config(params)
        inf_from_saved = resolve_from_saved_flags(
            getattr(params, "inf_from_saved", True),
            len(eval_tasks),
        )
        val_save_names = resolve_saved_workload_names(
            params.workload_profile,
            eval_tasks,
            "val",
            getattr(params, "inf_save_names", None),
        )
        test_save_names = resolve_saved_workload_names(
            params.workload_profile,
            eval_tasks,
            "test",
            getattr(params, "inf_save_names", None),
        )

        if params.run_mode == "ft":
            ######################################################################################################
            #                                          FINETUNE Task                                             #
            ######################################################################################################
            filter_func = data_size_filter

        else:
            ######################################################################################################
            #                                          Inference                                                 #
            ######################################################################################################
            filter_func = lambda x: x

        train_task = GOFAFineTuneTaskWrapper(train_tasks, root=params.data_root_path, split="train",
                                             hop=train_sampling["hops"],
                                             max_nodes_per_hop=train_sampling["train_max_nodes_per_hops"],
                                             sample_size=params.sample_size_per_task, filter_func=filter_func,
                                             way=params.ways, num_workers=params.num_workers,
                                             instruction=train_sampling["instructs"],
                                             selection=train_sampling["selections"], save_data=True,
                                             from_saved=True, fast_data_load=True)

        n_steps = int(len(train_task) * params.num_epochs / (params.grad_acc_step * int(torch.cuda.device_count())))
        val_tasks = [GOFAFineTuneTaskWrapper(task_name, root=params.data_root_path, split="val", hop=hop,
                                             max_nodes_per_hop=max_nodes_per_hop,
                                             sample_size=params.workload_profile["samples_per_split"],
                                             num_workers=params.num_workers, way=way, instruction=instruct,
                                             selection=selection, save_data=True, from_saved=from_saved,
                                             save_name=save_name) for
                     task_name, hop, max_nodes_per_hop, way, instruct, selection, from_saved, save_name in
                     zip(eval_tasks, params.inf_hops, params.inf_max_nodes_per_hops, params.inf_ways,
                         params.inf_instructs, params.inf_selections, inf_from_saved, val_save_names)]

        test_tasks = [GOFAFineTuneTaskWrapper(task_name, root=params.data_root_path, split="test", hop=hop,
                                              max_nodes_per_hop=max_nodes_per_hop, sample_size=inf_sample_size,
                                              num_workers=params.num_workers, way=way, instruction=instruct,
                                              selection=selection, save_data=True, from_saved=from_saved,
                                              save_name=save_name) for
                      task_name, hop, max_nodes_per_hop, way, instruct, selection, inf_sample_size, from_saved, save_name in
                      zip(eval_tasks, params.inf_hops, params.inf_max_nodes_per_hops, params.inf_ways,
                          params.inf_instructs, params.inf_selections, params.inf_sample_size_per_task,
                          inf_from_saved, test_save_names)]

        eval_metric_names, evaluators = get_evaluators(eval_tasks, task_types="QA")
        evlter = evaluators + evaluators

        train_task = DataWithMeta(train_task, batch_size=params.batch_size, sample_size=params.train_sample_size)
        val_tasks = [DataWithMeta(task, batch_size=params.batch_size, sample_size=params.eval_sample_size,
                                  state_name=task_name + "_val", metric=metric_name, classes=32132,
                                  meta_data={"eval_func": sentence_base}) for task_name, task, metric_name in
                     zip(eval_tasks, val_tasks, eval_metric_names)]

        test_tasks = [DataWithMeta(task, batch_size=params.batch_size, sample_size=params.eval_sample_size,
                                   state_name=task_name + "_test", metric=metric_name, classes=32132,
                                   meta_data={"eval_func": sentence_base}) for task_name, task, metric_name in
                      zip(eval_tasks, test_tasks, eval_metric_names)]

    text_dataset = {"train": train_task, "val": val_tasks, "test": test_tasks}
    params.datamodule = DataModule(text_dataset, num_workers=params.num_workers)

    ##################################################################
    #                    Setup model and optimizer                   #
    ##################################################################

    model = GOFA(
        transformer_args=[model_args, training_args, gofa_args],
        mode=params.mode,
        base_llm=params.base_llm,
        save_dir=params.exp_dir,
        print_generation_samples=getattr(params, "print_generation_samples", 0),
    )
    train_params = list(model.llm_model.model.icae.get_base_model().model.g_layers.parameters())
    if model_args.dec_lora:
        for name, param in model.llm_model.model.icae.named_parameters():
            if "default" in name and "lora" in name:
                train_params += [param]
    optimizer = torch.optim.AdamW(train_params, lr=params.lr, weight_decay=params.l2, betas=(0.9, 0.95))
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=params.lr * 0.1)
    lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
    # lr_scheduler_config = None

    ##################################################################
    #                    Setup evaluation and train                  #
    ##################################################################

    eval_data = text_dataset["val"] + text_dataset["test"]
    val_state = [dt.state_name for dt in text_dataset["val"]]
    test_state = [dt.state_name for dt in text_dataset["test"]]
    eval_state = val_state + test_state
    eval_metric = [dt.metric for dt in eval_data]
    eval_funcs = [dt.meta_data["eval_func"] for dt in eval_data]
    loss = torch.nn.CrossEntropyLoss()
    loss_func = normalized_loss_factory(params.batch_size, training_args.model_max_length)
    if len(evlter) == 0:
        for dt in eval_data:
            if dt.metric == "acc":
                evlter.append(Accuracy(task="multiclass", num_classes=dt.classes))
            elif dt.metric == "auc":
                evlter.append(AUROC(task="binary"))
            elif dt.metric == "apr":
                evlter.append(MultiApr(num_labels=dt.classes))
            elif dt.metric == "aucmulti":
                evlter.append(MultiAuc(num_labels=dt.classes))
            elif dt.metric.startswith("sim"):
                sim_metric = dt.metric.split("_")[1]
                evlter.append(SimAnyAuc(sim_metric))
            elif dt.metric == "loss":
                evlter.append(MeanMetric())
            elif dt.metric == "mae":
                evlter.append(MeanAbsoluteError())
            elif dt.metric == "perp":
                evlter.append(Perplexity(ignore_index=model.llm_model.model.tokenizer.pad_token_id))
            else:
                raise NotImplementedError("unknown evaluator")
    metrics = EvalKit(eval_metric, evlter, loss, eval_funcs, loss_func, eval_mode="max", exp_prefix="",
                      eval_state=eval_state, val_monitor_state=val_state[0], test_monitor_state=test_state[0], )

    exp_config = ExpConfig("", optimizer, lr_scheduler=lr_scheduler_config)
    exp_config.val_state_name = val_state
    exp_config.test_state_name = test_state
    pred_model = GraphTextPredLightning(exp_config, model, metrics)
    if params.load_model:
        print("-" * 60 + "LOADING" + "-" * 60)
        model.load_partial(load_dir=params.load_dir)
    strategy = "deepspeed_stage_2" if torch.cuda.device_count() > 1 else "auto"

    if params.run_mode == "inf":
        val_res, test_res = lightning_test(
            wandb_logger,
            pred_model,
            params.datamodule,
            metrics,
            strategy=strategy,
            run_validation=not params.skip_validation,
        )
    else:
        val_res, test_res = lightning_fit(wandb_logger, pred_model, params.datamodule, metrics,
                                          params.num_epochs + params.last_epochs, strategy=strategy,
                                          save_model=params.save_model["save"], load_best=False, reload_freq=1,
                                          val_interval=params.val_interval, grad_clipping=params.grad_clip,
                                          grad_acc_step=params.grad_acc_step,
                                          save_time=timedelta(hours=params.save_model["time"]), cktp_prefix="best_ckpt",
                                          precision=params.training_precision, top_k=params.save_model["top_k"],
                                          ckpt_path=params.ckpt_path, save_last=params.save_model["last"],
                                          ckpt_save_path=params.ckpt_save_path)
    if hasattr(model.llm_model, "_maybe_dump_encoder_cache_manifest"):
        model.llm_model._maybe_dump_encoder_cache_manifest(current_batch_seen=0, force=True)
    if hasattr(model.llm_model, "_maybe_dump_activation_observer"):
        model.llm_model._maybe_dump_activation_observer()
    if params.last_save:
        model.save_partial(os.path.join(params.exp_dir, "best_ckpt.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rl")
    parser.add_argument("--override", type=str)

    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line", )

    params = parser.parse_args()
    configs = []
    configs.append(load_yaml(os.path.join(os.path.dirname(__file__), "configs", "default_config.yaml")))

    if params.override is not None:
        override_config = load_yaml(params.override)
        configs.append(override_config)
    # Add for few-shot parameters

    mod_params = combine_dict(*configs)
    mod_params = merge_mod(mod_params, params.opts)
    mod_params["root_path"] = mod_params["root_path"] if mod_params["root_path"] else os.environ.get("GGAMA_ROOT_PATH")
    mod_params["data_root_path"] = mod_params["data_root_path"] if mod_params["data_root_path"] else os.environ.get(
        "GGAMA_ROOT_DATA_PATH")
    setup_exp(mod_params)

    params = SimpleNamespace(**mod_params)
    set_random_seed(params.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    torch.set_float32_matmul_precision("high")
    print(params)
    main(params)
