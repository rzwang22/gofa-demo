# LLM-N-SFT：纯 LLM-as-Predictor 基线

`LLM-N-SFT` 将 TAGLAS 已采样的 textual subgraph 线性化后，直接交给
`mistralai/Mistral-7B-Instruct-v0.2`。它不实例化 GOFA GNN、ICAE compressor、
memory tokens，也不读取 graph checkpoint。`model_type` 未设为 `llm_n` 时，原 GOFA
路径保持不变。

支持的实验任务为 `cora_node`、`cora_link`、`pubmed_node`、`pubmed_link`、
`arxiv`、`wikics`、`wn18rr`；同一实现也支持 `fb15k237` 和 `products`。

## 数据复用与防泄漏保证

LLM-N 直接调用 TAGLAS：

```text
get_task(name, task_type="QA", split, hop, max_nodes_per_hop, sample_size, ...)
  -> TAGLAS split / positive-negative samples / k-hop sampling / QA
  -> GOFA build_finetune_task_prompt(selection=False, instruction=False)
  -> deterministic LLM-N serialization
```

它使用与 GOFA 相同的 TAGLAS task cache（`save_data=true`、`from_saved=true`），不生成
新的 link negatives。序列化在单样本 `TAGData` 进入 PyG collate 之前完成，因此
`x[i]`、`edge_attr[e]` 和 sampled local index 不会被 batch remapping 混淆。

节点按“目标节点给定顺序优先，其余节点按全局 `node_map` 和 local index 排序”映射为
`[Node A]`、`[Node B]`……；边按 source、target、relation text、原 edge position 排序。
Cora-Link、PubMed-Link、WN18RR（以及扩展的 FB15K237）会在序列化前后断言目标 pair 的
任一方向均未出现在 edge/relation list。该断言用于检查 TAGLAS `LQATask` 的 target-edge
masking 是否仍然有效。

## 环境

按主 README 建立环境，并将 TAGLAS clone 到仓库根目录：

```bash
conda env create -f environment.yml
pip install -r requirements_llm_n.txt
git clone https://github.com/JiaruiFeng/TAGLAS.git
```

独立的 `requirements_llm_n.txt` 将 `accelerate` 固定为 0.26.1，供 Transformers 的
`low_cpu_mem_usage` / `device_map` 使用，同时不改动原 GOFA environment。LoRA SFT 使用本仓库内
隔离的 PyTorch 训练循环，不导入 Hugging Face Trainer。TAGLAS 本身尚未在原 GOFA 仓库中 pin
commit；正式对比时必须记录所用 TAGLAS commit，并确保其 PyTorch/PyG 版本与实验环境兼容。

## 训练

七任务联合 LoRA SFT（默认 `hops=3`、`max_nodes_per_hop=5`、batch size 1）。正式长任务建议从
一开始就固定 `run_name` 及 scheduler horizon；若使用 `max_steps`，首次启动时必须写最终目标值：

```bash
python run_llm_n.py --config configs/llm_n_train_config.yaml \
  run_name llm_n_7task_seed1 num_epochs 1 max_steps -1
```

仅训练一个或若干任务：

```bash
python run_llm_n.py --config configs/llm_n_train_config.yaml \
  --tasks cora_node cora_link
```

训练产物是标准 PEFT adapter 目录，不是 GOFA `.pth` graph checkpoint。也可以从兼容入口运行：

```bash
python run_gofa.py --override configs/llm_n_train_config.yaml
```

### 断点续训

从明确 checkpoint 恢复：

```bash
python run_llm_n.py --config configs/llm_n_train_config.yaml \
  --resume-from-checkpoint \
  outputs/llm_n_train/llm_n_7task_seed1/trainer/checkpoint-5000 \
  run_name llm_n_7task_seed1 num_epochs 1 max_steps -1
```

在指定 run 中自动选择 global step 最大的完整 checkpoint：

```bash
python run_llm_n.py --config configs/llm_n_train_config.yaml \
  --resume-from-checkpoint latest \
  --output-dir outputs/llm_n_train \
  run_name llm_n_7task_seed1 num_epochs 1 max_steps -1
```

也可在 YAML 中设置 `resume_from_checkpoint: /path/to/checkpoint` 或
`resume_from_checkpoint: latest`。恢复默认继续使用 checkpoint 所属的 `<run>`，不会创建新的 UTC
run id；`training_metrics.json`、`training_context_events.jsonl`、`trainer/checkpoint-*` 和
`adapter/` 都继续在原目录中更新。

恢复会先加载原始 Mistral，再以 `is_trainable=True` 加载 checkpoint 中的 PEFT adapter；随后恢复
optimizer、原 scheduler、FP16 GradScaler（如适用）、累计 loss/样本数、全部 RNG 状态及 sampler
cursor。sampler 的每个 epoch permutation 仅由 `seed + epoch` 决定，checkpoint cursor 指向下一个
尚未读取的样本，因此 `batch_size > 1`、`grad_acc_step > 1` 和 epoch 边界均不会重复或遗漏样本。

以下配置必须与 checkpoint 完全一致，否则程序会逐字段列出冲突并退出：base model、训练任务及
顺序、每任务 sample size、data root、seed、batch size、gradient accumulation、epoch/max steps、
hops/max nodes、全部 LoRA 参数、scheduler/warmup、precision、truncate mode，以及由这些配置解析
出的 dataset size 和 scheduler total steps。仅允许改变 `output_dir`、`run_name`、`logging_steps`、
`save_steps` 和 `save_total_limit`。特别地，不允许先按 `max_steps=2` 把 cosine scheduler 训练到 0，
再把同一 checkpoint 改为 `max_steps=100`。

### 推荐的正式训练命令

这里的“六任务”指七任务集合中去掉知识图谱任务 `wn18rr`，即 arXiv/Cora/PubMed/WikiCS 六项。
六任务：

```bash
python run_llm_n.py --config configs/llm_n_train_config.yaml \
  --tasks arxiv cora_node cora_link pubmed_node pubmed_link wikics \
  run_name llm_n_6task_seed1 \
  sample_size_per_task '[16000,1200,15000,1300,20000,8000]' \
  num_epochs 1 max_steps -1
```

七任务：

```bash
python run_llm_n.py --config configs/llm_n_train_config.yaml \
  run_name llm_n_7task_seed1 num_epochs 1 max_steps -1
```

以上命令按一个完整 epoch 训练，scheduler horizon 由首次解析出的数据集大小、batch size 和
gradient accumulation 决定。若改用正数 `max_steps`，正式实验应在首次启动前确定最终值，恢复命令
必须原样复用；`num_epochs` 和 `max_steps` 在两种方案下均不得在恢复时改变。

## 推理

零样本推理（不训练、不加载 adapter）：

```bash
python run_llm_n.py --config configs/llm_n_inference_config.yaml \
  --mode llm_n_zero_shot
```

对 LoRA SFT adapter 推理：

```bash
python run_llm_n.py --config configs/llm_n_inference_config.yaml \
  --mode llm_n_sft --adapter-path /absolute/path/to/adapter
```

七任务批量运行就是 inference config 的默认命令：

```bash
python run_llm_n.py --config configs/llm_n_inference_config.yaml
```

分别保存七个独立 run 时可使用：

```bash
for task in cora_node cora_link pubmed_node pubmed_link arxiv wikics wn18rr; do
  python run_llm_n.py --config configs/llm_n_inference_config.yaml \
    --tasks "$task" --output-dir "outputs/llm_n_7tasks/$task"
done
```

扩展任务：

```bash
python run_llm_n.py --config configs/llm_n_inference_config.yaml \
  --tasks fb15k237 products
```

## Smoke test

无需下载模型和数据的 synthetic smoke test 会对七个任务各运行 2 个样本，覆盖
serialization、forward、generate、answer-only labels、label parse、target-edge leakage
assertion 和 metric：

```bash
pytest -q tests/test_llm_n_serialization.py tests/test_llm_n_smoke.py
```

在完整 TAGLAS + CUDA 环境中执行真实 Mistral smoke：

```bash
python scripts/smoke_test_llm_n.py \
  --config configs/llm_n_inference_config.yaml \
  --output-dir outputs/llm_n_real_smoke
```

真实 Cora-node 断点恢复 smoke 从一开始设置 `max_steps=4`，使用 32 个训练样本，在原子完成
`checkpoint-2` 后中断，随后恢复到 step 4，并用最终 adapter 完成 20 个 test sample 推理：

```bash
python scripts/smoke_test_llm_n_resume.py \
  --output-dir outputs/llm_n_cora_resume_smoke
```

只检查当前环境是否具备 CUDA、TAGLAS、Transformers 和 PEFT：

```bash
python scripts/smoke_test_llm_n_resume.py --check-environment
```

无需 Mistral/TAGLAS 的恢复单元测试使用 tiny causal LM 和 synthetic dataset：

```bash
pytest -q tests/test_llm_n_resume.py
```

显存不足时，功能 smoke 可显式加 `--load-in-4bit`；量化 smoke 结果不能作为默认 BF16
公平对比结果。

## 输出格式

新运行在 `output_dir/<run_name 或 UTC run id>/` 下保存 resolved config；恢复运行始终复用 checkpoint
所属 run。推理目录为：

```text
<run>/
  resolved_config.json
  inference_summary.json
  <task>/<split>/metrics.json
  <task>/<split>/predictions.jsonl
```

训练另外保存 `training_metrics.json`、最终 `adapter/` 和按配置保留的
`trainer/checkpoint-<step>/`。checkpoint 的结构为：

```text
<run>/trainer/checkpoint-<step>/
  adapter_config.json
  adapter_model.safetensors       # 或 PEFT 所选的 adapter 权重格式
  tokenizer_config.json / tokenizer files
  training_state.pt               # optimizer/scheduler/scaler、cursor、累计量和 RNG
  training_config.json            # resolved config、关键配置和 SHA-256 fingerprint
  _SUCCESS                        # 仅完整 checkpoint 才存在
```

保存时先完整写入同级 `checkpoint-<step>.tmp/`，最后以 rename 提交为
`checkpoint-<step>/`；`latest` 只识别具有 `_SUCCESS`、state 和 config 的标准目录，忽略 `.tmp` 和
不完整目录。checkpoint 只会在 optimizer、scheduler step 和 `zero_grad` 完成后的 accumulation
边界生成。

显式截断会逐样本写入 `training_context_events.jsonl`；默认
`truncate_mode=none` 遇到超长训练样本时会先写入该文件和 `training_failure.json`，再终止训练。

`predictions.jsonl` 每行包含：

- `serialized_input`、`target_label`、`target_node_ids`；
- `raw_generation` 和 `normalized_prediction`；
- `parse_mode`、`parse_matched`、`correct`、`status`；
- `prefill_latency_ms`、`decode_latency_ms`、`total_latency_ms`；
- `original_input_token_count`、`input_token_count`、`output_token_count`；
- `truncated` 和显式 error（overflow/OOM 时）。

`metrics.json` 包含 accuracy、总/单样本 latency、prefill/decode latency、输入/输出 token
总数、各 GPU 与最大 peak allocated memory、truncation count、overflow count、OOM count，
以及 `overflow_oom_count`。overflow/OOM 样本以空预测进入 TAGLAS text_accuracy，因而计为错误；
latency per sample 的分母由 `num_latency_samples` 明确给出。

Cora/PubMed link 使用 TAGLAS 配置的 yes/no regex evaluator；WN18RR 只接受完整 canonical
relation label，再交给 TAGLAS exact-match evaluator。全类别表只用于生成后的 parser，绝不写入 prompt。

## Latency 与 context overflow

- 模型和数据构造完成后至少 warm up 10 次；warm-up 不计入结果。
- GPU prefill/decode 使用 `torch.cuda.Event`，每个阶段开始前和结束后 synchronize。
- `total_latency_ms = prefill_latency_ms + decode_latency_ms`，不含模型加载、数据下载和 tokenizer 时间。
- 默认 `truncate_mode: none`，绝不调用 tokenizer 静默截断。推理中的超长样本记录原 token 数、
  标为 `overflow` 并继续评测；训练中的超长样本先记录上下文事件，再 fail loudly，避免无声跳过样本。
- 只有显式配置 `truncate_mode: left|right|middle` 才会截断；每条被截断记录同时保存原始和
  截断后的 token 数（训练写 context events，推理写 predictions）。`middle` 会插入
  `[...TRUNCATED...]` 标记并确定性保留首尾 token。

## 与 GOFA 公平比较

至少固定并报告以下条件：

1. 同一个 TAGLAS checkout、`data_root_path` 和已保存 task cache。
2. 相同任务顺序、split、sample indices/size、seed、`hops=3`、`max_nodes_per_hop=5`。
3. GOFA supervised prompt 对应 `selection=False`、`instruction=False`；LLM-N 已强制此设置。
4. 相同 Mistral-7B-Instruct-v0.2 base、BF16、非量化、batch size 1 和 greedy decode token budget。
5. `truncate_mode=none`；overflow 单独报告，不能只在某一模型上丢弃超长样本。
6. 相同 GPU、CUDA/PyTorch/Transformers 版本；模型加载、dataset download 和前 10 次 warm-up
   均不计 latency。
7. 使用同一 TAGLAS `text_accuracy`，并同时报告 evaluated sample count、overflow/OOM count。

训练配置中的任务顺序和七个 `sample_size_per_task` 与 `configs/supervised_config.yaml` 对应项一致，
并默认设置 `gofa_size_filter: true`，复用 `run_gofa.py` 的 supervised sample filter。

## 七任务序列化示例

以下为缩短 node text 后的格式示例；真实运行会保留 sampled subgraph 的完整文本。prompt
总是止于 `Answer:`，正确答案只在 SFT labels 中出现。

### `cora_node`

```text
Task Description:
This is a Cora co-citation network. Predict the category of the target paper directly.

Target:
[Node A]

Nodes:
- [Node A]: Academic paper with title and abstract: Neural graph models ...
- [Node B]: Academic paper with title and abstract: Learning from citations ...

Edges:
- [Node A] -> [Node B]: Connected papers are cited together by other papers.

Question:
What is the most likely paper category for the paper?

Answer:
```

### `cora_link`

```text
Task Description:
Predict whether the two target Cora papers are co-cited. Generate Yes or No directly.

Target:
[Node A] and [Node B]

Nodes:
- [Node A]: Academic paper with title and abstract: Paper one ...
- [Node B]: Academic paper with title and abstract: Paper two ...
- [Node C]: Academic paper with title and abstract: Shared context ...

Edges:
- [Node A] -> [Node C]: Connected papers are cited together by other papers.
- [Node C] -> [Node B]: Connected papers are cited together by other papers.

Question:
Is two papers co-cited or not? Please answer yes if co-cited and no otherwise.

Answer:
```

目标 `[Node A]`–`[Node B]` 边不在 edge list 中。

### `pubmed_node`

```text
Task Description:
Predict the category of the target PubMed paper directly.

Target:
[Node A]

Nodes:
- [Node A]: PubMed paper with title and abstract: Diabetes study ...
- [Node B]: PubMed paper with title and abstract: Insulin response ...

Edges:
- [Node B] -> [Node A]: This paper cites the other paper.

Question:
What is the most likely paper category for the paper?

Answer:
```

### `pubmed_link`

```text
Task Description:
Predict whether the two target PubMed papers are co-cited. Generate Yes or No directly.

Target:
[Node A] and [Node B]

Nodes:
- [Node A]: PubMed paper with title and abstract: Target study one ...
- [Node B]: PubMed paper with title and abstract: Target study two ...
- [Node C]: PubMed paper with title and abstract: Neighbor study ...

Edges:
- [Node A] -> [Node C]: The papers are connected in the citation graph.
- [Node C] -> [Node B]: The papers are connected in the citation graph.

Question:
Is two target papers co-cited or not? Please answer yes or no.

Answer:
```

目标 `[Node A]`–`[Node B]` 边不在 edge list 中。

### `arxiv`

```text
Task Description:
Predict the arXiv category of the target paper directly.

Target:
[Node A]

Nodes:
- [Node A]: Academic paper title and abstract: Scalable representation learning ...
- [Node B]: Academic paper title and abstract: Optimization for deep networks ...

Edges:
- [Node A] -> [Node B]: The source paper cites the target paper.

Question:
What is the most likely paper category for the paper?

Answer:
```

### `wikics`

```text
Task Description:
Predict the category of the target Wikipedia term directly.

Target:
[Node A]

Nodes:
- [Node A]: Wikipedia entry: Artificial neural network ...
- [Node B]: Wikipedia entry: Machine learning ...

Edges:
- [Node B] -> [Node A]: The source page links to the target page.

Question:
What is the most likely category for the Wikipedia term?

Answer:
```

### `wn18rr`

```text
Task Description:
Predict the canonical relation between the two target WordNet entities directly.

Target:
[Node A] and [Node B]

Nodes:
- [Node A]: English word and description: dog ...
- [Node B]: English word and description: animal ...
- [Node C]: English word and description: canine ...

Edges:
- [Node A] -> [Node C]: Relation from source word to target word: _synset_domain_topic_of
- [Node C] -> [Node B]: Relation from source word to target word: _hypernym

Question:
What is the relationship between two target words?

Answer:
```

目标 `[Node A]`–`[Node B]` relation 不在 relation list 中。

## 已知限制

- 原仓库没有 pin TAGLAS commit；TAGLAS main 当前要求的 PyTorch 版本可能高于原 GOFA
  `environment.yml`。不要在未验证的版本组合间比较结果。
- 当前 runner 的性能统计按单样本 greedy decode 实现；尚未提供 batched latency benchmark 或
  sampling/beam search。
- LoRA SFT 使用可确定性恢复的单进程 PyTorch loop；目前未提供分布式 sampler、FSDP 或
  DeepSpeed checkpoint 恢复。
- 公平 latency benchmark 应让模型完整放在一张 GPU 上。`device_map=auto` 可用于功能推理，但若
  它把模型真正切到多张 GPU，当前以输入设备 CUDA Event 记录的 latency 不应作为跨模型公平数据。
