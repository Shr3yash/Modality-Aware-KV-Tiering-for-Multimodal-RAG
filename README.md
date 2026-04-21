# Modality-Policy-for-Multimodal-RAG


GPU requirement: 80GB at least

pipeline:
→ datasetName_dataset.py: organize the data from dataset (file) to map or list for code
→ retriever.py: compare the query with corpus and retrieve the top-k
→ prune.py: simple pruning applied
→ prompt_builder.py: build a prompt with instructions.
→ query_pipeline.py: enter the rendered prompt into the MLLM and generate output.
→ eval_baseline.py: take the output from query pipeline and evaluate
→ output in output folder


Before starting, install necessary packages `lmcache` and `vllm`.


Load dataset:
```
./scripts/download_mmdocrag.sh
```

Start the vllm server
```
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --max-model-len 16384 \
  --gpu_memory_utilization 0.8 \ # add this line to reduce GPU footprint
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

In another terminal:
```
PYTHONPATH=$PYTHONPATH:. python scripts/run_mmdocrag_baseline.py
```

Reset used disk:
```
rm -rf ~/lmcache_storage
```