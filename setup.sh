curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash

export VLLM_METAL_USE_MLX=1 # turn on framework for Apple Silicon
export VLLM_MLX_DEVICE=gpu  # gpu

source ~/.venv-vllm-metal/bin/activate
pip install -r requirements.txt