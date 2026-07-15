# GOFA Trace Source Audit

This document is a Static Audit + Runtime Validation Plan. It is not a final query trace exporter and does not define the final simulator trace schema.

Agent did not execute GOFA on the target datasets. Runtime semantics marked as RUNTIME_VALIDATION_REQUIRED must be confirmed using the generated audit commands.

Static audit source revision at drafting time: `52e16646d7fadb5418c086a3d7963efef2e70c46`. Runtime audit files also write the actual `repository_commit_sha` observed on the server.

## Scope

The final goal is a trace-driven GOFA workload for PIM+NPU analytical replay:

- Layer 1, Static Dataset/Model Metadata: dataset identity, verified global namespace, graph counts/degrees, model dimensions, memory-token count, suffix-layer count, GNN-layer count, sampler configuration.
- Layer 2, Query Structural Trace: real query target, NOG, actual sampled graph, local indices, verified local/global mapping, structural `edge_index`, edge text-item mapping.
- Layer 3, Runtime Access/Execution Trace: selected K/V item masks, effective consumed K/V, complete K/V item sets, selected token counts, suffix-layer sharing, QK/PV/GNN shapes.

Current implementation work only audits source-field semantics. It does not implement the three-layer formal exporter, global-degree policy, or a new cache design.

Current research design is quantized memory cache, quantized text-side K/V cache, selective text-side K/V loading, W4A8 integer Linear, quantized-KV attention, INT QK, and INT PV. Existing names such as `load_key_base`, `load_value_base`, and `kv_base_load_policy` are legacy implementation names for text-side K/V loading controls.

## Status Legend

- `STATIC_VERIFIED`: confirmed from definition and use sites.
- `RUNTIME_VALIDATION_REQUIRED`: static evidence exists, but real task runs must confirm concrete values and offsets.
- `VERIFIED_BY_RUNTIME`: reserved for future reports produced after the user supplies real audit JSON. This status is not assigned to any field in this first audit.
- `UNRESOLVED`: current code and available local environment are insufficient.

## Runtime Probe

Implemented default-off config:

```yaml
gofa_trace_source_audit:
  enabled: false
  output_dir: ""
  max_queries: 1
  include_full_arrays: true
  flush_interval: 1
  rank_zero_only: true
  strict: false
  dump_pre_cache_snapshot: false
  dump_cache_miss_snapshot: true
```

Flat CLI fields are also supported:

```text
gofa_trace_source_audit_enabled
gofa_trace_source_audit_output_dir
gofa_trace_source_audit_max_queries
gofa_trace_source_audit_include_full_arrays
gofa_trace_source_audit_flush_interval
gofa_trace_source_audit_rank_zero_only
gofa_trace_source_audit_strict
gofa_trace_source_audit_dump_pre_cache_snapshot
gofa_trace_source_audit_dump_cache_miss_snapshot
```

When enabled, `dump_pre_cache_snapshot` writes the item mapping before any cache lookup. On a strict
quant cache miss, `dump_cache_miss_snapshot` atomically writes `cache_miss_*.json` before preserving
the original `RuntimeError`. Both diagnostics are metadata-only and do not export cache tensors.

Code evidence:

- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:99): `ModelArguments` fields.
- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:1733): audit metadata writer.
- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:2131): query snapshot builder.
- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:3931): hook point after cache reconstruction and K/V mask application, before debug zero, ablation, assembly, and `forward_memory_with_text_kv`.

Output is:

```text
<output_dir>/
  audit_metadata.json
  query_000000.json
  audit.log
```

The probe writes raw index arrays, shapes, counts, mapping summaries, item type/ranges, selected masks, and small metadata only. It does not write hidden states, K/V tensor values, weights, or activation tensors.

## Field Audit

### `graph.target_index`

Status: RUNTIME_VALIDATION_REQUIRED.

Definition evidence:

- [tasks/pretrain_tasks.py](/Users/wangrunze/Desktop/gofa-demo/tasks/pretrain_tasks.py:179): node task receives `target_index` from `__process_graph__`.
- [tasks/pretrain_tasks.py](/Users/wangrunze/Desktop/gofa-demo/tasks/pretrain_tasks.py:202): task wrapper writes `target_index` into `TAGData`.
- [tasks/pretrain_task_base.py](/Users/wangrunze/Desktop/gofa-demo/tasks/pretrain_task_base.py:90): downstream task wraps `target_index.tolist()`.
- [tasks/pretrain_task_base.py](/Users/wangrunze/Desktop/gofa-demo/tasks/pretrain_task_base.py:523): link-prediction pretrain task appends `edge.tolist()` as target endpoints.
- [tasks/task_base.py](/Users/wangrunze/Desktop/gofa-demo/tasks/task_base.py:116): fine-tune graph construction uses `data.target_index` to build question/NOG prompt edges.

Use evidence:

- [modules/gofa/cache_policy.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/cache_policy.py:158): `_target_local_indices` filters flattened `target_index` into `[0, num_node_feat)`.
- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:3374): Scheme-B ablation and policy helpers also interpret `target_index` as sampled local node indices.

Static conclusion:

- Static code confirms model-side Scheme-B policies require local node indices, not full-graph ids.
- Static code suggests link tasks can carry two node endpoints, not an edge id, when built by `LinkPrediction`.
- Static code does not prove every `cora_node`, `cora_link`, `pubmed_node`, and `wikics` runtime sample enters the model with the same shape after TAGLAS sampling and batching.

Runtime questions:

- Is `target_index` always sampled-graph local after TAGLAS `__process_graph__` for all requested datasets?
- Does batch size 2 increment `target_index` via TAGLAS/PyG collate?
- Does `target_index` ever include a NOG/prompt node after `build_GOFA_task_graph`?

Probe fields: `target_index.raw`, `num_node_feat`, `all_values_in_num_node_feat_range`, `node_map_at_target_index`, `question_index_raw`, and `candidate_nog_local_index`.

### `graph.node_map`

Status: STATIC_VERIFIED for current model use as cache-item reorder indices; UNRESOLVED as a global-node-id mapping.

Definition evidence:

- [tasks/pretrain_tasks.py](/Users/wangrunze/Desktop/gofa-demo/tasks/pretrain_tasks.py:221): task build collects node text from upstream `data.node_map`.
- [tasks/pretrain_tasks.py](/Users/wangrunze/Desktop/gofa-demo/tasks/pretrain_tasks.py:245): `np.unique(..., return_inverse=True)` creates `unique_node_map`.
- [tasks/pretrain_tasks.py](/Users/wangrunze/Desktop/gofa-demo/tasks/pretrain_tasks.py:251): `data_list[i].node_map` is overwritten with the inverse map into unique node text items.

Use evidence:

- [modules/gofa/gofa_modeling.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa_modeling.py:979): `item_order = graph.node_map.tolist() + edge item range`.
- [modules/gofa/gofa_modeling.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa_modeling.py:980): memory states are reordered by `item_order`.

Static conclusion:

- In this code path, `node_map` is consumed as an encoder/cache item index map for sampled node slots.
- Because it is rewritten from text de-duplication, it cannot be called a verified global node id reference.

Runtime questions:

- Does any independent field preserve original/global node id in saved task samples?
- Does `node_map` remain a valid permutation or inverse map after batch size 2 collation?

Probe fields: raw values, min/max, uniqueness, permutation check, valid encoder item index check, values at target/question indices.

### `graph.edge_map`

Status: STATIC_VERIFIED for current model use as structural-edge to edge-text-item offset; UNRESOLVED as global edge id.

Definition evidence:

- [tasks/task_base.py](/Users/wangrunze/Desktop/gofa-demo/tasks/task_base.py:71): prompt edges are appended to `edge_index`.
- [tasks/task_base.py](/Users/wangrunze/Desktop/gofa-demo/tasks/task_base.py:72): existing `edge_map` is read.
- [tasks/task_base.py](/Users/wangrunze/Desktop/gofa-demo/tasks/task_base.py:75): prompt edge map is appended with an offset.
- [tasks/task_base.py](/Users/wangrunze/Desktop/gofa-demo/tasks/task_base.py:78): prompt edge text is appended to `edge_attr`.

Use evidence:

- [modules/gofa/gofa_modeling.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa_modeling.py:991): `gnn_edge_input = memory_states[cur_node_size:][graph.edge_map]`.
- [modules/gofa/cache_policy.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/cache_policy.py:167): `_edge_item_index` maps structural edge position to cache item `num_node_feat + edge_map_value`.

Static conclusion:

- Formal traces must store structural `edge_index`, `edge_map`, and edge text-item inventory separately.
- The code does not guarantee a one-to-one relation between structural edges and edge text items.

Runtime questions:

- How often do multiple structural edges share one edge text item for each dataset/task?
- Are prompt/NOG incident edges represented in `edge_map` after batch collation?

Probe fields: length, unique count, min/max, edge text item count, multiplicity histogram, valid edge-text-item index check.

### NOG and `question_index`

Status: RUNTIME_VALIDATION_REQUIRED.

Definition evidence:

- [tasks/task_base.py](/Users/wangrunze/Desktop/gofa-demo/tasks/task_base.py:29): `construct_prompt_graph` creates prompt/NOG nodes after original nodes.
- [tasks/task_base.py](/Users/wangrunze/Desktop/gofa-demo/tasks/task_base.py:42): prompt node local index is `num_nodes + i`.
- [tasks/task_base.py](/Users/wangrunze/Desktop/gofa-demo/tasks/task_base.py:121): fine-tune multi-node question stores prompt node in `question_index`.
- [tasks/task_base.py](/Users/wangrunze/Desktop/gofa-demo/tasks/task_base.py:132): without prompt graph, `question_index` can equal `target_index`.
- [tasks/task_base.py](/Users/wangrunze/Desktop/gofa-demo/tasks/task_base.py:168): pretrain no-prompt path can mix prompt node indices and single target indices.

Use evidence:

- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:3087): `_encoder_cache_skip_indices` checks `question_index`.
- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:3099): skip cache item is computed through `graph.node_map[question_index]`.
- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:3508): ablation skips online/NOG cache items when zeroing cached nodes.

Static conclusion:

- NOG/question nodes are first-class graph nodes in prompt graph construction when prompt graph is added.
- Cache skip uses `question_index` through `node_map`, so runtime validation must confirm local index and cache item mapping.

Runtime questions:

- Does NOG participate in `graph.edge_index` and GNN message passing for each task?
- Does `encoder_cache_skip_nog=True` exclude all prompt/NOG cache items and only those items?

Probe fields: question raw, candidate local nodes, node_map values at question, skip cache indices, incident/in/out edge counts, persistent eligibility, GNN participation.

### Edge-Side Text K/V

Status: STATIC_VERIFIED for node-item then edge-item text order at model input; RUNTIME_VALIDATION_REQUIRED for per-task edge reuse.

Use evidence:

- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:4107): `text_inputs` concatenates `g.x` then `g.edge_attr`.
- [modules/gofa/gofa_modeling.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa_modeling.py:979): edge cache item range is appended after mapped node item order.
- [modules/gofa/gofa_modeling.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa_modeling.py:999): suffix Transformer loops every mapped item.
- [modules/gofa/gofa_modeling.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa_modeling.py:1016): each item consumes its layer K/V entry.

Static conclusion:

- Edge items can carry memory state and suffix text-side K/V if they are part of `token_ids`.
- GNN reads edge memory via `edge_map`; suffix Transformer traverses edge items as items.
- `keep_target_edges` and `kv_base_keep_target_edges` select edge cache items incident to selected local nodes via `edge_map`.

Runtime questions:

- Do all edge items in the sampled graph have complete suffix-layer K/V in the quant cache?
- How often do edge text items get reused by multiple structural edges?

### Selected K Mask and Selected V Mask

Status: STATIC_VERIFIED for mask construction; RUNTIME_VALIDATION_REQUIRED for effective runtime attention behavior under K-only/V-only audit configs.

Definition and use evidence:

- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:3242): `_build_scheme_b_kv_base_load_masks` receives eligible cache items.
- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:3290): `load_key_base` gates selected key items.
- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:3291): `load_value_base` gates selected value items.
- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:3292): `complete_kv_selected` is intersection of key and value selected sets.
- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:3774): quant reconstruct receives key/value loads gated by `complete_kv_base_load_mask`.
- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:3326): loaded cache item is emptied when complete K/V is not retained.

Static conclusion:

- Policy K and V masks can differ in config, but the first implementation only feeds complete K/V items into attention.
- K-only or V-only settings are audit stress tests, not supported production semantics.

Probe fields: policy selected K/V indices, effective consumed K/V indices, complete K/V, K-only, V-only, eligible items, skip items.

### Suffix Layer Selection

Status: STATIC_VERIFIED for shared item-level mask, RUNTIME_VALIDATION_REQUIRED by output arrays.

Evidence:

- [modules/gofa/gofa_modeling.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa_modeling.py:985): suffix loop covers `layers[gnn_start_layer:num_hidden_layers]`.
- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:3242): one K/V item policy is built per batch before suffix forward.
- [modules/gofa/gofa.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/gofa.py:3921): policy is applied once to cache item payloads before suffix forward.

Static conclusion:

- Current selection is item-level and shared across suffix layers.
- The runtime probe still emits effective K/V item arrays per suffix layer.

### Degree Semantics

Status: STATIC_VERIFIED for sampled-local degree; UNRESOLVED for global graph degree.

Evidence:

- [modules/gofa/cache_policy.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/cache_policy.py:88): `local_node_degrees` counts only `graph.edge_index`.
- [modules/gofa/cache_policy.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/cache_policy.py:200): `_local_degree_top_nodes` selects from sampled graph degree.
- [modules/gofa/cache_policy.py](/Users/wangrunze/Desktop/gofa-demo/modules/gofa/cache_policy.py:265): `target_1hop_local_degree` uses sampled-local degree selection.

Static conclusion:

- Current `local_degree_top` is sampled-local degree. Future papers/traces should call it `sampled_local_degree_top`.
- Global high-degree policy should be based on verified `global_graph_degree`, which is not implemented here.

### Batch Offset and Collation

Status: UNRESOLVED until server runtime audit.

Evidence:

- [tasks/task_wrapper.py](/Users/wangrunze/Desktop/gofa-demo/tasks/task_wrapper.py:135): wrapper delegates batching to `self.task_list[0].collate(batch)`.
- [gp/lightning/data_template.py](/Users/wangrunze/Desktop/gofa-demo/gp/lightning/data_template.py:83): `DatasetWithCollate` path uses the task collate function.
- [gp/lightning/data_template.py](/Users/wangrunze/Desktop/gofa-demo/gp/lightning/data_template.py:94): PyG dataset path uses PyG `DataLoader` with dataset collate.

Local environment cannot import `TAGLAS`, so the exact `TAGData.__inc__` / `__cat_dim__` behavior is not statically available. Runtime audit must compare batch size 1 and 2 for `target_index`, `question_index`, `edge_index`, `node_map`, `edge_map`, `batch`, and `ptr`.
