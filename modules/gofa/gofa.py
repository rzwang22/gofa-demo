# example code for running inference with fine-tuned checkpoint
from typing import Optional

import hashlib
import json
import os
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from transformers import MistralConfig

from modules.gofa.gofa_icae import MistralICAE
from collections import OrderedDict
from safetensors.torch import load_file
from modules.utils import safe_download_hf_file

###################################################################
#                 Configurations                                  #
###################################################################


class GOFAMistralConfig(MistralConfig):
    def __init__(self, dim=4096, num_layers=6, mem_token=128, head=32, add_self_loops=True, dropout=0.0,
                 llama_dtype=torch.float16, gnn_hidden_act="relu", gnn_mlp_type="gp", gnn_type="index", position_encoding="none", pretraining_tp=0, gating=True, interleave=True, mp_att="concat", trainable_layer=5, fuse_type="interleave", **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.mem_token = mem_token
        self.head = head
        self.add_self_loops = add_self_loops
        self.dropout = dropout
        self.num_layers = num_layers
        self.llama_dtype = llama_dtype
        self.gnn_hidden_act = gnn_hidden_act
        self.gnn_mlp_type = gnn_mlp_type
        self.gnn_type = gnn_type
        self.pretraining_tp = pretraining_tp
        self.position_encoding = position_encoding
        self.interleave = interleave
        self.gating = gating
        self.mp_att = mp_att
        self.trainable_layer = trainable_layer
        self.fuse_type = fuse_type


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    attn_implementation: str = field(default="eager", metadata={"help": "Mistral attention implementation"})
    lora_r: int = field(default=512, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    mem_size: int = field(default=128, metadata={"help": "Memory size"}, )
    dec_lora: bool = field(default=False, metadata={"help": "Whether using lora in the decoder LLM"})
    checkpoint_dir: str = field(default="./cache_data/model/")
    use_encoder_cache: bool = field(default=False, metadata={"help": "Cache graph-independent encoder prefix states"})
    encoder_cache_dir: str = field(default="./cache_data/encoder_cache")
    encoder_cache_tag: str = field(default="")
    encoder_cache_skip_nog: bool = field(default=False, metadata={"help": "Do not cache NOG/prompt node states"})


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    bf16: bool = field(default=False)
    model_max_length: int = field(default=512,
        metadata={"help": "Maximum sequence length per node. Sequences will be right padded (and possibly truncated)."}, )
    fixed_mem_size: int = field(default=128, metadata={"help": "Enalbing the fixed mem size."}, )
    mean_compression_rate: int = field(default=4, metadata={"help": "Mean compression rate; default=4"}, )
    min_tokens_for_lm: int = field(default=64, metadata={"help": "Minimum tokens for lm objective learning"}, )
    leave_tokens_for_lm: int = field(default=8, metadata={"help": "Leave some tokens without loss for lm objective"}, )
    lm_ratio: float = field(default=0.0, metadata={"help": "Ratio for LM training."}, )
    add_special_token_for_lm: bool = field(default=False,
        metadata={"help": "Add a special token for the prompt of language modeling; default: False"}, )
    restore_from: str = field(default="",
        metadata={"help": "The checkpoint that should be restored from for fine-tuning"})


###################################################################
#                 Model                                           #
###################################################################


class GOFAMistral(torch.nn.Module):
    def __init__(self, transformer_args):
        super().__init__()
        model_args, training_args, gofa_args = transformer_args
        model = MistralICAE(model_args, training_args, gofa_args)  # restored llama2-7b-chat model
        dir = safe_download_hf_file("sggetao/icae", "mistral_7b_ft_icae.safetensors", model_args.checkpoint_dir, repo_type=None)
        state_dict = load_file(dir)  # change the path for your model
        new_state_dict = OrderedDict()
        for layer_name, weight in state_dict.items():
            new_state_dict[layer_name.replace("default", "encadapt")] = weight
        model.load_state_dict(new_state_dict, strict=False)
        # model.merge_lora()
        self.model_args = model_args
        self.dec_lora = model_args.dec_lora
        self.mem_tokens = list(range(model.vocab_size, model.vocab_size + model_args.mem_size))
        self.mem_size = model_args.mem_size
        self.model = model
        self.encoder_cache_enabled = bool(model_args.use_encoder_cache)
        self.encoder_cache_dir = model_args.encoder_cache_dir
        self.encoder_cache_calls = 0
        self.encoder_cache_hits = 0
        self.encoder_cache_misses = 0
        self.encoder_cache_skips = 0
        self.encoder_cache_timing = {
            "load_s": 0.0,
            "miss_compute_s": 0.0,
            "save_s": 0.0,
            "assemble_s": 0.0,
            "suffix_compute_s": 0.0,
            "total_s": 0.0,
        }
        self.encoder_full_calls = 0
        self.encoder_full_time_s = 0.0
        self.encoder_cache_namespace = self._build_encoder_cache_namespace(dir, model_args, training_args, gofa_args)
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.model.left_tokenizer.pad_token = self.model.left_tokenizer.bos_token
        for param in self.model.icae.parameters():
            param.requires_grad = False
        for param in self.model.icae.get_base_model().model.g_layers.parameters():
            param.requires_grad = True
        if self.dec_lora:
            for name, param in self.model.icae.named_parameters():
                if "default" in name:
                    param.requires_grad = True
        if self.encoder_cache_enabled:
            if gofa_args.fuse_type != "interleave":
                print("GOFA encoder cache is only implemented for fuse_type=interleave; disabling cache.")
                self.encoder_cache_enabled = False
            else:
                os.makedirs(self._encoder_cache_root(), exist_ok=True)
                print(
                    "GOFA encoder cache enabled: "
                    f"dir={self._encoder_cache_root()}, boundary_layer="
                    f"{self.model.icae.get_base_model().model.gnn_start_layer}, "
                    f"skip_nog={model_args.encoder_cache_skip_nog}"
                )

    def get_tokenizer(self):
        return self.model.tokenizer

    def train_mode(self):
        self.model.icae.set_adapter("encadapt")
        for param in self.model.icae.parameters():
            param.requires_grad = False

    def load_pretrained(self, pretrained_path=None):
        if pretrained_path is None:
            pretrained_path = safe_download_hf_file("WFRaain/GOFA", "mistral_qamag03_best_ckpt.pth", self.model_args.checkpoint_dir,
                                        repo_type=None)
        self.load_partial(pretrained_path)

    def save_partial(self, save_dir):
        """
        Save the GNN and lora weight (if available).
        """
        state_dict = self.model.icae.get_base_model().model.g_layers.state_dict()
        full_state_dict = self.state_dict()
        for k in full_state_dict:
            if "default" in k:
                state_dict[k] = full_state_dict[k]
        torch.save(state_dict, save_dir)

    def load_partial(self, load_dir):
        """
        Load the GNN and lora weight (if available).
        """
        state_dict = torch.load(load_dir, map_location="cpu")
        normalized_state_dict = OrderedDict()
        for key, value in state_dict.items():
            normalized_state_dict[key] = value
            if key.startswith("llm_model."):
                normalized_state_dict[key[len("llm_model."):]] = value

        raw_lora_keys = [key for key in state_dict if "default" in key and "lora" in key.lower()]
        loadable_lora_keys = [
            key for key in normalized_state_dict
            if not key.startswith("llm_model.") and "default" in key and "lora" in key.lower()
        ]
        print(
            "Loaded partial checkpoint keys: "
            f"total={len(state_dict)}, normalized={len(normalized_state_dict)}, "
            f"raw_decoder_lora={len(raw_lora_keys)}, loadable_decoder_lora={len(loadable_lora_keys)}"
        )
        missing_keys, unexpected_keys = self.model.icae.get_base_model().model.g_layers.load_state_dict(state_dict, strict=False)
        print("GNN module is missing the following keys:", missing_keys)
        if unexpected_keys:
            print("GNN module skipped non-GNN keys:", len(unexpected_keys))
        missing_keys, unexpected_keys = self.load_state_dict(normalized_state_dict, strict=False)
        if self.dec_lora:
            missing_lora = [key for key in missing_keys if "default" in key and "lora" in key.lower()]
            print("Decoder LoRA keys in checkpoint:", len(raw_lora_keys))
            print("Decoder LoRA keys after prefix normalization:", len(loadable_lora_keys))
            print("Decoder LoRA keys still missing after load:", len(missing_lora))
            if not loadable_lora_keys:
                raise RuntimeError("dec_lora=True, but no loadable decoder LoRA keys were found in the checkpoint.")
            if missing_lora:
                raise RuntimeError("dec_lora=True, but decoder LoRA keys were not fully loaded.")
        if unexpected_keys:
            print("Full GOFA module skipped unexpected keys:", len(unexpected_keys))

    def _build_encoder_cache_namespace(self, icae_path, model_args, training_args, gofa_args):
        stat = os.stat(icae_path)
        base_config = self.model.icae.get_base_model().config
        metadata = {
            "format": 1,
            "model_name_or_path": model_args.model_name_or_path,
            "model_max_length": training_args.model_max_length,
            "mem_size": model_args.mem_size,
            "vocab_size": self.model.vocab_size,
            "hidden_size": self.model.dim,
            "num_hidden_layers": base_config.num_hidden_layers,
            "num_gnn_layers": gofa_args.num_layers,
            "fuse_type": gofa_args.fuse_type,
            "attn_implementation": model_args.attn_implementation,
            "bf16": training_args.bf16,
            "icae_path": os.path.abspath(icae_path),
            "icae_size": stat.st_size,
            "icae_mtime_ns": stat.st_mtime_ns,
            "base_model_fingerprint": self._local_model_fingerprint(model_args.model_name_or_path),
            "tag": model_args.encoder_cache_tag,
        }
        payload = json.dumps(metadata, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _local_model_fingerprint(self, model_path):
        if not os.path.exists(model_path):
            return None
        if os.path.isfile(model_path):
            stat = os.stat(model_path)
            return [{"path": os.path.basename(model_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}]

        fingerprint = []
        for name in sorted(os.listdir(model_path)):
            if not (name.endswith(".json") or name.endswith(".safetensors") or name.endswith(".bin")):
                continue
            path = os.path.join(model_path, name)
            if not os.path.isfile(path):
                continue
            stat = os.stat(path)
            fingerprint.append({"path": name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        return fingerprint

    def _encoder_cache_root(self):
        return os.path.join(self.encoder_cache_dir, self.encoder_cache_namespace)

    def _encoder_cache_key(self, token_ids):
        token_bytes = np.asarray(token_ids, dtype=np.int32).tobytes()
        return hashlib.sha256(token_bytes).hexdigest()

    def _encoder_cache_path(self, cache_key):
        return os.path.join(self._encoder_cache_root(), cache_key[:2], cache_key + ".pt")

    def _load_encoder_cache_item(self, token_ids):
        cache_key = self._encoder_cache_key(token_ids)
        cache_path = self._encoder_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return None, cache_key
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("cache_key") != cache_key or payload.get("seq_len") != len(token_ids):
            return None, cache_key
        return payload["hidden_state"], cache_key

    def _save_encoder_cache_item(self, cache_key, token_ids, hidden_state):
        cache_path = self._encoder_cache_path(cache_key)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        payload = {
            "cache_key": cache_key,
            "seq_len": len(token_ids),
            "dtype": str(hidden_state.dtype),
            "hidden_state": hidden_state.detach().cpu(),
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)

    def _sync_encoder_cache_timer(self, device):
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)

    def _encoder_cache_log_timing(self, current):
        total_cacheable_items = self.encoder_cache_hits + self.encoder_cache_misses
        total_items = total_cacheable_items + self.encoder_cache_skips
        hit_rate = self.encoder_cache_hits / total_cacheable_items if total_cacheable_items else 0.0
        skip_rate = self.encoder_cache_skips / total_items if total_items else 0.0
        timing = self.encoder_cache_timing
        print(
            "GOFA encoder cache timing: "
            f"call_total={self.encoder_cache_calls}, hit_rate={hit_rate:.2%}, "
            f"skip_rate={skip_rate:.2%}, "
            f"current_total={current['total_s']:.4f}s, "
            f"current_load={current['load_s']:.4f}s, "
            f"current_miss_compute={current['miss_compute_s']:.4f}s, "
            f"current_save={current['save_s']:.4f}s, "
            f"current_assemble={current['assemble_s']:.4f}s, "
            f"current_suffix={current['suffix_compute_s']:.4f}s, "
            f"cum_total={timing['total_s']:.4f}s, "
            f"cum_load={timing['load_s']:.4f}s, "
            f"cum_miss_compute={timing['miss_compute_s']:.4f}s, "
            f"cum_save={timing['save_s']:.4f}s, "
            f"cum_assemble={timing['assemble_s']:.4f}s, "
            f"cum_suffix={timing['suffix_compute_s']:.4f}s"
        )

    def _encoder_cache_skip_indices(self, graph):
        if not self.model_args.encoder_cache_skip_nog or graph is None:
            return []
        if not hasattr(graph, "question_index") or not hasattr(graph, "node_map"):
            return []
        question_index = graph.question_index
        if question_index is None or question_index.numel() == 0:
            return []
        question_index = question_index.to(graph.node_map.device)
        valid_mask = (question_index >= 0) & (question_index < graph.node_map.numel())
        if not torch.any(valid_mask):
            return []
        skip_indices = torch.unique(graph.node_map[question_index[valid_mask]]).detach().cpu().tolist()
        return [int(idx) for idx in skip_indices if 0 <= int(idx) < graph.num_node_feat]

    def _encode_with_encoder_cache(
            self,
            token_ids,
            padded_token_ids,
            mem_mask,
            graph=None,
            partial_grad=None,
            skip_cache_indices=None):
        if self.training:
            return None
        base_model = self.model.icae.get_base_model().model
        if not hasattr(base_model, "forward_llm_prefix") or not hasattr(base_model, "forward_from_gnn_boundary"):
            return None

        device = padded_token_ids.device
        self._sync_encoder_cache_timer(device)
        total_start = time.perf_counter()
        current_timing = {
            "load_s": 0.0,
            "miss_compute_s": 0.0,
            "save_s": 0.0,
            "assemble_s": 0.0,
            "suffix_compute_s": 0.0,
            "total_s": 0.0,
        }

        skip_cache_indices = set(skip_cache_indices or [])
        cached_states = [None] * len(token_ids)
        missing = []
        missing_keys = []
        skipped = []
        load_start = time.perf_counter()
        for i, ids in enumerate(token_ids):
            if i in skip_cache_indices:
                missing.append(i)
                missing_keys.append(None)
                skipped.append(i)
                continue
            cached_state, cache_key = self._load_encoder_cache_item(ids)
            if cached_state is None:
                missing.append(i)
                missing_keys.append(cache_key)
            else:
                cached_states[i] = cached_state
        current_timing["load_s"] = time.perf_counter() - load_start

        if missing:
            self._sync_encoder_cache_timer(device)
            miss_compute_start = time.perf_counter()
            missing_token_ids = [token_ids[i] for i in missing]
            missing_padded = self.model.tokenizer.pad(
                {"input_ids": missing_token_ids}, padding=True, return_tensors="pt"
            )["input_ids"].to(padded_token_ids.device)
            missing_embeddings = self.model.tokens_to_embeddings(missing_padded)
            prefix_output = base_model.forward_llm_prefix(
                inputs_embeds=missing_embeddings,
                partial_grad=partial_grad,
                return_dict=True,
            ).last_hidden_state
            self._sync_encoder_cache_timer(device)
            current_timing["miss_compute_s"] = time.perf_counter() - miss_compute_start

            save_start = time.perf_counter()
            for batch_idx, original_idx in enumerate(missing):
                seq_len = len(token_ids[original_idx])
                cached_state = prefix_output[batch_idx, :seq_len].detach().cpu()
                cached_states[original_idx] = cached_state
                if missing_keys[batch_idx] is not None:
                    self._save_encoder_cache_item(missing_keys[batch_idx], token_ids[original_idx], cached_state)
            current_timing["save_s"] = time.perf_counter() - save_start

        self._sync_encoder_cache_timer(device)
        assemble_start = time.perf_counter()
        hidden_dtype = cached_states[0].dtype
        boundary_hidden_states = torch.zeros(
            (len(cached_states), padded_token_ids.size(1), cached_states[0].size(-1)),
            dtype=hidden_dtype,
            device=padded_token_ids.device,
        )
        for i, cached_state in enumerate(cached_states):
            seq_len = cached_state.size(0)
            boundary_hidden_states[i, :seq_len] = cached_state.to(padded_token_ids.device)
        self._sync_encoder_cache_timer(device)
        current_timing["assemble_s"] = time.perf_counter() - assemble_start

        self.encoder_cache_calls += 1
        current_skips = len(skipped)
        current_misses = len(missing) - current_skips
        current_hits = len(token_ids) - len(missing)
        self.encoder_cache_hits += current_hits
        self.encoder_cache_misses += current_misses
        self.encoder_cache_skips += current_skips

        self._sync_encoder_cache_timer(device)
        suffix_start = time.perf_counter()
        final_hidden_states = base_model.forward_from_gnn_boundary(
            boundary_hidden_states=boundary_hidden_states,
            graph=graph,
            mem_mask=mem_mask,
            partial_grad=partial_grad,
            map_node=True,
            return_dict=True,
        ).last_hidden_state
        self._sync_encoder_cache_timer(device)
        current_timing["suffix_compute_s"] = time.perf_counter() - suffix_start
        current_timing["total_s"] = time.perf_counter() - total_start

        for key, value in current_timing.items():
            self.encoder_cache_timing[key] += value

        if self.encoder_cache_calls <= 3 or self.encoder_cache_calls % 50 == 0:
            print(
                "GOFA encoder cache: "
                f"call={self.encoder_cache_calls}, hits={current_hits}, "
                f"misses={current_misses}, skips={current_skips}, "
                f"total_hits={self.encoder_cache_hits}, total_misses={self.encoder_cache_misses}, "
                f"total_skips={self.encoder_cache_skips}"
            )
            self._encoder_cache_log_timing(current_timing)

        return final_hidden_states

    def forward(self, g):
        """
        Encode the graph and generate logits for answer tokens.
        """
        g.num_node_feat = g.x.shape[0]
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            text_inputs = np.concatenate([g.x, g.edge_attr], axis=0)
        else:
            text_inputs = g.x
        text_inputs = text_inputs.tolist()
        llm_output = self.encode(text_inputs, graph=g, partial_grad=True)
        emb = llm_output[:g.node_map.size(-1)]
        if not hasattr(g, "answer"):
            raise ValueError("Forward stage graph should contain answer.")
        answer_texts = g.answer[g.answer_map.cpu().numpy()].tolist()
        prompt_texts = g.question[g.question_map.cpu().numpy()].tolist()
        # Legacy hard coding TODO: remove when TAGLAS is fixed.
        prompt_input_texts = ["" if (p.startswith("Please complete the sentence of the node") or p == "") else p for p
                              in prompt_texts]
        emb = emb[g.question_index]
        answer_logits, answer_id, masks = self.decode(answer_texts, emb, prompt=prompt_input_texts)
        return answer_logits, answer_id, masks, answer_texts

    def generate(self, g, max_length=128):
        """
        Autoregressively generate tokens.
        """
        g.num_node_feat = g.x.shape[0]
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            text_inputs = np.concatenate([g.x, g.edge_attr], axis=0)
        else:
            text_inputs = g.x
        text_inputs = text_inputs.tolist()
        llm_output = self.encode(text_inputs, graph=g, partial_grad=True)
        emb = llm_output[:g.node_map.size(-1)]
        prompt_texts = g.question[g.question_map.cpu().numpy()].tolist()
        prompt_input_texts = ["" if (p.startswith("Please complete the sentence of the node") or p == "") else p for p
                              in prompt_texts]
        emb = emb[g.question_index]
        generated_text = self.infer(emb, prompt=prompt_input_texts, max_length=max_length)
        return generated_text

    def encode(self, data, graph=None, partial_grad=None):
        cur_device = self.model.memory_token_embed.weight.device
        batch_size = len(data)
        text_output = \
        self.model.tokenizer(data, truncation=True, max_length=self.model.training_args.model_max_length, padding=False,
                             return_attention_mask=False)["input_ids"]

        text_output = [t + self.mem_tokens for t in text_output]
        padded_text_output = {"input_ids": text_output}
        padded_text_output = self.model.tokenizer.pad(padded_text_output, padding=True, return_tensors="pt")["input_ids"].to(
            cur_device)
        mem_mask = padded_text_output >= self.model.vocab_size

        mem_mask = mem_mask.to(cur_device)

        # Use ICAE lora only in the encoder.
        self.model.icae.set_adapter("encadapt")
        self.model.icae.enable_adapter_layers()
        for name, param in self.model.icae.named_parameters():
            if "encadapt" in name:
                param.requires_grad = False
        compress_outputs = None
        if self.encoder_cache_enabled and graph is not None:
            compress_outputs = self._encode_with_encoder_cache(
                text_output,
                padded_text_output,
                mem_mask,
                graph=graph,
                partial_grad=partial_grad,
                skip_cache_indices=self._encoder_cache_skip_indices(graph),
            )
        if compress_outputs is None:
            self._sync_encoder_cache_timer(cur_device)
            full_start = time.perf_counter()
            autoencoder_input_embedding = self.model.tokens_to_embeddings(padded_text_output)
            compress_outputs = self.model.icae(inputs_embeds=autoencoder_input_embedding, output_hidden_states=True,
                                               graph=graph, mem_mask=mem_mask, partial_grad=partial_grad, map_node=True)
            compress_outputs = compress_outputs.hidden_states[-1]
            self._sync_encoder_cache_timer(cur_device)
            full_elapsed = time.perf_counter() - full_start
            self.encoder_full_calls += 1
            self.encoder_full_time_s += full_elapsed
            if self.encoder_full_calls <= 3 or self.encoder_full_calls % 50 == 0:
                print(
                    "GOFA encoder full path timing: "
                    f"call={self.encoder_full_calls}, "
                    f"current={full_elapsed:.4f}s, "
                    f"cum_total={self.encoder_full_time_s:.4f}s"
                )
        self.model.icae.disable_adapter_layers()

        if graph is not None:
            node_emb = compress_outputs[:len(graph.node_map)]
            map_mem_mask = mem_mask[:graph.num_node_feat][graph.node_map]
            memory_embedding = node_emb[map_mem_mask].view(len(node_emb), self.mem_size, -1)
        else:
            memory_embedding = compress_outputs[mem_mask].view(batch_size, self.mem_size, -1)
        return memory_embedding

    def decode(self, data, mem_embs, graph=None, prompt=None):
        prompt_output = self.model.tokenizer(data, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        prompt_output = [p + [self.model.tokenizer.eos_token_id] if len(p) < self.model.training_args.model_max_length else p[:self.model.training_args.model_max_length] for p in prompt_output]
        original_prompt_output = prompt_output

        if prompt is None:
            prompt = [""] * len(data)
        prompt_input = self.model.left_tokenizer(prompt, add_special_tokens=False, padding=False, truncation=True, max_length=512)["input_ids"]
        batch_size = len(prompt_input)

        # For Mistral, decode contains: prefix, memory slots and suffix
        prompt_left_ids = [[1, 733, 16289, 28793] if len(a) > 0 else [] for a in prompt_input]
        prompt_right_ids = [[self.model.ft_token_id] + a + [733, 28748, 16289, 28793] if len(a) > 0 else a for a in
                            prompt_input]
        prompt_ids = [a + [self.model.tokenizer.pad_token_id] * self.mem_size + b + c for a, b, c in
                      zip(prompt_left_ids, prompt_right_ids, prompt_output)]
        prompt_mask = [
            [False] * (len(prompt_left_ids[i]) + self.mem_size - 1 + len(prompt_right_ids[i])) + [True] * len(
                prompt_output[i]) + [False] for i in range(batch_size)]

        answer_prompt = torch.cat([torch.tensor(p, dtype=torch.long) for p in prompt_output], dim=-1).to(
            mem_embs.device)

        prompt_output = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
        prompt_output = self.model.tokenizer.pad(prompt_output, padding=True, return_tensors="pt")
        prompt_answer_ids = prompt_output["input_ids"].to(mem_embs.device)
        prompt_answer_embs = self.model.tokens_to_embeddings(prompt_answer_ids)

        mem_mask = [[False] * len(prompt_left_ids[i]) + [True] * self.mem_size + [False] * (
                len(prompt_output["input_ids"][i]) - len(prompt_left_ids[i]) - self.mem_size) for i in
                    range(batch_size)]
        prompt_mask = [
            [False] * (len(prompt_left_ids[i]) + self.mem_size - 1 + len(prompt_right_ids[i])) + [True] * len(
                original_prompt_output[i]) + [False] * (1 + len(prompt_output["input_ids"][i]) - len(prompt_ids[i])) for
            i in range(batch_size)]

        prompt_answer_embs[torch.tensor(mem_mask)] = mem_embs.view(-1, mem_embs.size()[-1])

        target_mask = torch.tensor(prompt_mask, dtype=torch.long, device=mem_embs.device).to(torch.bool)

        if self.dec_lora:
            self.model.icae.set_adapter("default")
            self.model.icae.enable_adapter_layers()
        else:
            self.model.icae.disable_adapter_layers()
        output_emb = self.model.icae(inputs_embeds=prompt_answer_embs).logits

        return output_emb, answer_prompt, target_mask

    def infer(self, mem_embs, graph=None, prompt=None, max_length=128):
        cur_device = self.model.memory_token_embed.weight.device

        if prompt is None:
            prompt = [""] * len(mem_embs)
        prompt_input = self.model.tokenizer(prompt, add_special_tokens=False, padding=False)["input_ids"]
        batch_size = len(prompt_input)

        prompt_left_ids = [[1, 733, 16289, 28793] if len(a) > 0 else [] for a in prompt_input]

        prompt_right_ids = [[self.model.ft_token_id] + a + [733, 28748, 16289, 28793] if len(a) > 0 else a for a in
                            prompt_input]

        mem_mask = [[False] * len(prompt_left_ids[i]) + [True] * self.mem_size + [False] * len(prompt_right_ids[i]) for
                    i in range(batch_size)]
        att_mask = [[True] * (len(prompt_left_ids[i]) + self.mem_size + len(prompt_right_ids[i])) for i in
                    range(batch_size)]
        prompt_ids = [prompt_left_ids[i] + [self.model.tokenizer.pad_token_id] * self.mem_size + prompt_right_ids[i] for
                      i in range(batch_size)]

        input_prompt_ids = self.model.left_tokenizer.pad({"input_ids": prompt_ids, "attention_mask": mem_mask},
                                                         padding=True, return_tensors="pt")
        mem_mask = input_prompt_ids["attention_mask"].to(device=mem_embs.device, dtype=torch.bool)

        input_prompt_ids = self.model.left_tokenizer.pad({"input_ids": prompt_ids, "attention_mask": att_mask},
                                                         padding=True, return_tensors="pt")
        prompt_ids = input_prompt_ids["input_ids"]
        att_mask = input_prompt_ids["attention_mask"].to(device=mem_embs.device)

        prompt_answer_ids = prompt_ids.to(device=mem_embs.device, dtype=torch.long)
        prompt_answer_embs = self.model.tokens_to_embeddings(prompt_answer_ids)
        prompt_answer_embs[mem_mask] = mem_embs.view(-1, mem_embs.size()[-1])

        decode_embed = prompt_answer_embs
        output = decode_embed.clone()

        generate_text = []
        eos_reached = torch.zeros(len(output), dtype=torch.bool).to(output.device)

        past_key_values = None
        if self.dec_lora:
            self.model.icae.set_adapter("default")
            self.model.icae.enable_adapter_layers()
        else:
            self.model.icae.disable_adapter_layers()
        for i in range(max_length):
            out = self.model.icae(inputs_embeds=output, attention_mask=att_mask, past_key_values=past_key_values,
                                 use_cache=True)

            logits = out.logits[:, -1, :self.model.vocab_size - 1]

            past_key_values = out.past_key_values

            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            eos_reached = torch.logical_or(eos_reached, (next_token_id == self.model.tokenizer.eos_token_id).view(-1))

            # eos_reached = torch.logical_or(eos_reached, (next_token_id==self.model.tokenizer.bos_token_id).view(-1))

            # eos_reached = torch.logical_or(eos_reached, (next_token_id>=32000).view(-1))

            output = self.model.icae.get_base_model().model.embed_tokens(next_token_id).to(mem_embs.device)

            generate_text.append(next_token_id.view(-1, 1))
            att_mask = torch.cat(
                [att_mask, torch.ones((len(att_mask), 1), dtype=att_mask.dtype, device=att_mask.device)], dim=-1)

            if torch.all(eos_reached):
                break

        generate_text = torch.cat(generate_text, dim=-1)
        generate_text[generate_text >= 32000] = 1

        generated_text = self.model.tokenizer.batch_decode(generate_text)

        return generated_text
