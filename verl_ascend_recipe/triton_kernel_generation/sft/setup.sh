#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

# clone verl (if not exists)
if [ ! -d "${ROOT_DIR}/verl" ]; then
    git clone https://github.com/verl-project/verl.git "${ROOT_DIR}/verl"
else
    echo "[INFO] verl already exists, skip clone"
fi

cd "${ROOT_DIR}/verl"
# ensure correct commit
VERL_COMMIT="c780fc34b45e01a1538d6386947585d4f7370bef"
git fetch origin
git checkout ${VERL_COMMIT}

pip install -e . --no-build-isolation --no-deps

pip install --no-cache-dir "ray==2.47.1"

pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org
pip install torch-npu==2.8.0

pip install transformers==4.57.6
pip install tensordict==0.10.0
pip install einops==0.8.2
pip install peft==0.18.1
pip install datasets==4.8.4
pip install codetiming==1.4.0
pip install pybind11==3.0.2
pip install pylatexenc==2.10
pip install tensorboard==2.20.0
pip install wandb==0.25.1
pip install torchdata==0.11.0
pip install sandbox-fusion
pip install logfire
pip install gradio
pip install huggingface_hub==0.36.2
pip install protobuf==3.20
pip install hydra-core
pip install numpy==1.26.0

# Install flash-attn-2.8.3
ABI_FLAG="${ABI_FLAG:-FALSE}"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI_FLAG}-cp310-cp310-linux_x86_64.whl"
wget -nv -P . "${URL}"
pip install --no-cache-dir "./$(basename "${URL}")"