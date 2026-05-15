# Modality-Policy-for-Multimodal-RAG

GPU requirement: H200 GPU, 140GB

pipeline:
→ datasetName_dataset.py: organize the data from dataset (file) to map or list for code
→ retriever.py: compare the query with corpus and retrieve the top-k
→ prune.py: simple pruning applied
→ prompt_builder.py: build a prompt with instructions.
→ query_pipeline.py: enter the rendered prompt into the MLLM and generate output.
→ eval_baseline.py: take the output from query pipeline and evaluate
→ output in output folder

Before starting, install conda. Then
```
conda env create -f environment.yml
```

Activate the environment:
```
conda activate mrag
```

Load dataset:

```
./scripts/download_mmdocrag.sh
```

Start the vllm server

```
PYTHONPATH=$PYTHONPATH:. \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --max-model-len 16384 \
  --gpu_memory_utilization=0.5 \
  --host 0.0.0.0 \
  --port 8000

```

In another terminal:

```
# JSONL lines 10–19 (10 rows), no extra cap
PYTHONPATH=$PYTHONPATH:. python scripts/run_mmdocrag_baseline.py --eval-slice-start 0 --eval-slice-stop 50 --max-examples 0

# From line 100, at most 15 examples
PYTHONPATH=$PYTHONPATH:. python scripts/run_mmdocrag_baseline.py --eval-slice-start 100 --max-examples 15

```

