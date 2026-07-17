"""Mistral + LoRA implementation for the pure LLM-N predictor."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .serialization import SerializedGraphSample


VALID_TRUNCATE_MODES = frozenset({"none", "left", "right", "middle"})


class ContextOverflowError(RuntimeError):
    def __init__(self, original_tokens: int, capacity: int, context: str = "prompt") -> None:
        self.original_tokens = int(original_tokens)
        self.capacity = int(capacity)
        self.context = context
        super().__init__(
            f"{context} has {original_tokens} tokens but only {capacity} are allowed; "
            "set truncate_mode explicitly to left, right, or middle to permit truncation"
        )


@dataclass(frozen=True)
class TokenizedPrompt:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    original_token_count: int
    token_count: int
    truncated: bool


@dataclass(frozen=True)
class TokenizedTrainingSample:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_token_count: int
    answer_token_count: int
    original_token_count: int
    token_count: int
    truncated: bool


@dataclass(frozen=True)
class GenerationResult:
    raw_text: str
    generated_token_ids: tuple[int, ...]
    input_token_count: int
    original_input_token_count: int
    output_token_count: int
    truncated: bool
    prefill_latency_ms: float
    decode_latency_ms: float
    total_latency_ms: float


def _cfg(config: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


class LLMNTokenizer:
    """Tokenization policy with explicit overflow handling and answer masking."""

    def __init__(
        self,
        tokenizer: Any,
        max_context_length: int = 32768,
        max_new_tokens: int = 32,
        truncate_mode: str = "none",
    ) -> None:
        if truncate_mode not in VALID_TRUNCATE_MODES:
            raise ValueError(
                f"truncate_mode must be one of {sorted(VALID_TRUNCATE_MODES)}, got {truncate_mode!r}"
            )
        if max_context_length <= 0:
            raise ValueError("max_context_length must be positive")
        if max_new_tokens <= 0 or max_new_tokens >= max_context_length:
            raise ValueError("max_new_tokens must be positive and smaller than max_context_length")
        self.tokenizer = tokenizer
        self.max_context_length = int(max_context_length)
        self.max_new_tokens = int(max_new_tokens)
        self.truncate_mode = truncate_mode

    @property
    def pad_token_id(self) -> int:
        value = getattr(self.tokenizer, "pad_token_id", None)
        if value is None:
            value = getattr(self.tokenizer, "eos_token_id", None)
        if value is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
        return int(value)

    @property
    def eos_token_id(self) -> int | None:
        value = getattr(self.tokenizer, "eos_token_id", None)
        return None if value is None else int(value)

    def _encode(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        return [int(token) for token in encoded]

    def format_prompt(self, serialized_prompt: str) -> str:
        """Apply Mistral's chat template while keeping ``Answer:`` assistant-side."""
        marker = "\nAnswer:"
        if serialized_prompt.endswith(marker):
            user_content = serialized_prompt[:-len(marker)]
        elif serialized_prompt.rstrip().endswith("Answer:"):
            user_content = serialized_prompt.rstrip()[:-len("Answer:")].rstrip()
        else:
            user_content = serialized_prompt
        messages = [{"role": "user", "content": user_content}]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, TypeError, ValueError):
            rendered = f"<s>[INST] {user_content} [/INST]"
        return rendered.rstrip() + "\nAnswer: "

    def _truncate(self, token_ids: list[int], capacity: int, context: str) -> tuple[list[int], bool]:
        if len(token_ids) <= capacity:
            return token_ids, False
        if self.truncate_mode == "none":
            raise ContextOverflowError(len(token_ids), capacity, context)
        if capacity <= 0:
            raise ContextOverflowError(len(token_ids), capacity, context)
        if self.truncate_mode == "left":
            return token_ids[-capacity:], True
        if self.truncate_mode == "right":
            return token_ids[:capacity], True

        marker_ids = self._encode("\n[...TRUNCATED...]\n")
        if len(marker_ids) >= capacity:
            return marker_ids[:capacity], True
        remaining = capacity - len(marker_ids)
        prefix_count = remaining // 2
        suffix_count = remaining - prefix_count
        suffix = token_ids[-suffix_count:] if suffix_count else []
        return token_ids[:prefix_count] + marker_ids + suffix, True

    def encode_prompt(self, serialized_prompt: str) -> TokenizedPrompt:
        model_prompt = self.format_prompt(serialized_prompt)
        original_ids = self._encode(model_prompt)
        capacity = self.max_context_length - self.max_new_tokens
        input_ids, truncated = self._truncate(original_ids, capacity, "inference prompt")
        return TokenizedPrompt(
            input_ids=tuple(input_ids),
            attention_mask=tuple(1 for _ in input_ids),
            original_token_count=len(original_ids),
            token_count=len(input_ids),
            truncated=truncated,
        )

    def encode_training_sample(self, sample: SerializedGraphSample) -> TokenizedTrainingSample:
        prompt_ids = self._encode(self.format_prompt(sample.prompt))
        answer_ids = self._encode(sample.answer.strip())
        eos_token_id = self.eos_token_id
        if eos_token_id is not None and (not answer_ids or answer_ids[-1] != eos_token_id):
            answer_ids.append(eos_token_id)
        if not answer_ids:
            raise ValueError("The answer must contain at least one token")
        prompt_capacity = self.max_context_length - len(answer_ids)
        original_total = len(prompt_ids) + len(answer_ids)
        try:
            prompt_ids, truncated = self._truncate(
                prompt_ids,
                prompt_capacity,
                "training prompt",
            )
        except ContextOverflowError as error:
            # Attach sample identity and total counts so the training runner can
            # persist a useful failure record before failing loudly.
            error.task_name = sample.task_name
            error.sample_index = sample.sample_index
            error.original_total_token_count = original_total
            error.answer_token_count = len(answer_ids)
            error.max_context_length = self.max_context_length
            raise
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        if any(label != -100 for label in labels[:len(prompt_ids)]):
            raise AssertionError("Prompt labels must all be -100")
        return TokenizedTrainingSample(
            input_ids=tuple(input_ids),
            attention_mask=tuple(1 for _ in input_ids),
            labels=tuple(labels),
            prompt_token_count=len(prompt_ids),
            answer_token_count=len(answer_ids),
            original_token_count=original_total,
            token_count=len(input_ids),
            truncated=truncated,
        )


class AnswerOnlyDataCollator:
    """Right-pad samples while masking every prompt and padding label."""

    def __init__(
        self,
        tokenizer_policy: LLMNTokenizer,
        include_metadata: bool = False,
    ) -> None:
        self.tokenizer_policy = tokenizer_policy
        self.include_metadata = bool(include_metadata)

    def __call__(self, features: Sequence[SerializedGraphSample]) -> dict[str, Any]:
        import torch

        encoded = [self.tokenizer_policy.encode_training_sample(feature) for feature in features]
        max_length = max(item.token_count for item in encoded)
        pad_token_id = self.tokenizer_policy.pad_token_id
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for item in encoded:
            padding = max_length - item.token_count
            input_ids.append(list(item.input_ids) + [pad_token_id] * padding)
            attention_mask.append(list(item.attention_mask) + [0] * padding)
            labels.append(list(item.labels) + [-100] * padding)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if self.include_metadata:
            # This key is consumed by the isolated training loop and is never
            # forwarded to the causal LM.  It makes every explicit training
            # truncation auditable without changing the regular collator API.
            batch["_llm_n_metadata"] = [
                {
                    "task_name": feature.task_name,
                    "sample_index": feature.sample_index,
                    "original_token_count": item.original_token_count,
                    "token_count": item.token_count,
                    "prompt_token_count": item.prompt_token_count,
                    "answer_token_count": item.answer_token_count,
                    "truncated": item.truncated,
                }
                for feature, item in zip(features, encoded)
            ]
        prompt_mask = batch["labels"].eq(-100)
        if not bool(prompt_mask.any()) or not bool(batch["labels"].ne(-100).any()):
            raise AssertionError("Each SFT batch must contain masked prompt and unmasked answer tokens")
        return batch


class LLMNPredictor:
    """Pure causal-LM predictor with an optional PEFT LoRA adapter."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_context_length: int = 32768,
        max_new_tokens: int = 32,
        truncate_mode: str = "none",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenization = LLMNTokenizer(
            tokenizer,
            max_context_length=max_context_length,
            max_new_tokens=max_new_tokens,
            truncate_mode=truncate_mode,
        )

    @classmethod
    def from_pretrained(cls, config: Mapping[str, Any] | Any, for_training: bool = False) -> "LLMNPredictor":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = _cfg(config, "model_name_or_path", "mistralai/Mistral-7B-Instruct-v0.2")
        dtype_name = str(_cfg(config, "torch_dtype", "bfloat16")).lower()
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "auto": "auto",
        }
        if dtype_name not in dtype_by_name:
            raise ValueError(f"Unsupported torch_dtype {dtype_name!r}")
        torch_dtype = dtype_by_name[dtype_name]
        if not torch.cuda.is_available() and torch_dtype in (torch.float16, torch.bfloat16):
            torch_dtype = torch.float32

        resume_checkpoint = _cfg(config, "_resolved_resume_checkpoint", None)
        tokenizer_source = resume_checkpoint if for_training and resume_checkpoint else model_name
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            use_fast=bool(_cfg(config, "use_fast_tokenizer", True)),
            trust_remote_code=bool(_cfg(config, "trust_remote_code", False)),
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": bool(_cfg(config, "trust_remote_code", False)),
            "low_cpu_mem_usage": True,
        }
        attention_implementation = _cfg(config, "attn_implementation", None)
        if attention_implementation:
            model_kwargs["attn_implementation"] = attention_implementation
        device_map = _cfg(config, "device_map", None)
        if device_map:
            model_kwargs["device_map"] = device_map

        load_in_4bit = bool(_cfg(config, "load_in_4bit", False))
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        configured_context = int(_cfg(config, "max_context_length", 32768))
        model_context = getattr(model.config, "max_position_embeddings", None)
        if model_context is not None and configured_context > int(model_context):
            raise ValueError(
                f"max_context_length={configured_context} exceeds model limit {model_context}"
            )

        adapter_path = _cfg(config, "adapter_path", None)
        if for_training:
            if adapter_path:
                raise ValueError(
                    "adapter_path is inference-only; use resume_from_checkpoint for trainable LoRA recovery"
                )
            if load_in_4bit:
                from peft import prepare_model_for_kbit_training

                model = prepare_model_for_kbit_training(model)
            if resume_checkpoint:
                from peft import PeftModel

                model = PeftModel.from_pretrained(
                    model,
                    resume_checkpoint,
                    is_trainable=True,
                )
            else:
                from peft import LoraConfig, get_peft_model

                target_modules = _cfg(
                    config,
                    "lora_target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj"],
                )
                lora_config = LoraConfig(
                    r=int(_cfg(config, "lora_r", 16)),
                    lora_alpha=int(_cfg(config, "lora_alpha", 32)),
                    lora_dropout=float(_cfg(config, "lora_dropout", 0.05)),
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=list(target_modules),
                )
                model = get_peft_model(model, lora_config)
            if bool(_cfg(config, "gradient_checkpointing", True)):
                model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
            model.config.use_cache = False
            trainable_names = [
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            ]
            if not trainable_names or any("lora_" not in name for name in trainable_names):
                raise RuntimeError(
                    "Training model must expose only LoRA parameters as trainable; got: "
                    + ", ".join(trainable_names)
                )
            model.train()
            model.print_trainable_parameters()
        elif adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            model.config.use_cache = True

        return cls(
            model=model,
            tokenizer=tokenizer,
            max_context_length=configured_context,
            max_new_tokens=int(_cfg(config, "max_new_tokens", 32)),
            truncate_mode=str(_cfg(config, "truncate_mode", "none")),
        )

    @property
    def device(self) -> Any:
        if hasattr(self.model, "get_input_embeddings"):
            embeddings = self.model.get_input_embeddings()
            weight = getattr(embeddings, "weight", None)
            if weight is not None and getattr(weight.device, "type", None) != "meta":
                return weight.device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            import torch

            return torch.device("cpu")

    def forward(self, **batch: Any) -> Any:
        return self.model(**batch)

    def generate(self, sample: SerializedGraphSample | str) -> GenerationResult:
        """Greedy autoregressive generation with separate prefill/decode events."""
        import torch

        prompt = sample.prompt if isinstance(sample, SerializedGraphSample) else str(sample)
        tokenized = self.tokenization.encode_prompt(prompt)
        device = self.device
        input_ids = torch.tensor([tokenized.input_ids], dtype=torch.long, device=device)
        attention_mask = torch.tensor([tokenized.attention_mask], dtype=torch.long, device=device)
        use_cuda_events = bool(torch.cuda.is_available() and getattr(device, "type", str(device)) == "cuda")

        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        generated: list[int] = []
        eos_token_id = self.tokenization.eos_token_id
        forward_parameters = inspect.signature(self.model.forward).parameters.values()
        generation_forward_kwargs: dict[str, Any] = {}
        if (
            "num_logits_to_keep" in inspect.signature(self.model.forward).parameters
            or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in forward_parameters)
        ):
            # Transformers 4.48 Mistral otherwise materializes [prompt, vocab]
            # logits even though generation needs only the final position.
            generation_forward_kwargs["num_logits_to_keep"] = 1

        with torch.inference_mode():
            if use_cuda_events:
                torch.cuda.synchronize(device)
                cuda_stream = torch.cuda.current_stream(device=device)
                prefill_start = torch.cuda.Event(enable_timing=True)
                prefill_end = torch.cuda.Event(enable_timing=True)
                prefill_start.record(cuda_stream)
            else:
                prefill_wall_start = time.perf_counter()

            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                **generation_forward_kwargs,
            )

            if use_cuda_events:
                prefill_end.record(cuda_stream)
                torch.cuda.synchronize(device)
                prefill_ms = float(prefill_start.elapsed_time(prefill_end))
                torch.cuda.synchronize(device)
                decode_start = torch.cuda.Event(enable_timing=True)
                decode_end = torch.cuda.Event(enable_timing=True)
                decode_start.record(cuda_stream)
            else:
                prefill_ms = (time.perf_counter() - prefill_wall_start) * 1000.0
                decode_wall_start = time.perf_counter()

            next_logits = output.logits[:, -1, :]
            past_key_values = getattr(output, "past_key_values", None)
            for token_position in range(self.tokenization.max_new_tokens):
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                token_id = int(next_token.item())
                generated.append(token_id)
                if eos_token_id is not None and token_id == eos_token_id:
                    break
                if token_position + 1 >= self.tokenization.max_new_tokens:
                    break
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                    dim=-1,
                )
                output = self.model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    **generation_forward_kwargs,
                )
                next_logits = output.logits[:, -1, :]
                past_key_values = getattr(output, "past_key_values", past_key_values)

            if use_cuda_events:
                decode_end.record(cuda_stream)
                torch.cuda.synchronize(device)
                decode_ms = float(decode_start.elapsed_time(decode_end))
            else:
                decode_ms = (time.perf_counter() - decode_wall_start) * 1000.0

        if was_training:
            self.model.train()
        raw_text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return GenerationResult(
            raw_text=raw_text,
            generated_token_ids=tuple(generated),
            input_token_count=tokenized.token_count,
            original_input_token_count=tokenized.original_token_count,
            output_token_count=len(generated),
            truncated=tokenized.truncated,
            prefill_latency_ms=prefill_ms,
            decode_latency_ms=decode_ms,
            total_latency_ms=prefill_ms + decode_ms,
        )
