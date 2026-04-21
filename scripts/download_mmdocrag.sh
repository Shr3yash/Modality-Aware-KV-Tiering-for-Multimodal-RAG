#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/mmdocrag
cd data/mmdocrag

# annotations
wget -O dev_15.jsonl \
  https://raw.githubusercontent.com/MMDocRAG/MMDocRAG/main/dataset/dev_15.jsonl

wget -O evaluation_15.jsonl \
  https://raw.githubusercontent.com/MMDocRAG/MMDocRAG/main/dataset/evaluation_15.jsonl

# image quotes zip from Hugging Face dataset assets
wget -O images.zip \
  https://huggingface.co/datasets/MMDocIR/MMDocRAG/resolve/main/images.zip

mkdir -p images
unzip -o images.zip -d images