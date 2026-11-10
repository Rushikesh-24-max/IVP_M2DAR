#!/usr/bin/env bash
# Run inside a dedicated Python 3.10 or 3.11 virtual environment on the GPU server.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python -c 'import sys; assert sys.version_info[:2] in ((3, 10), (3, 11)), "Use Python 3.10 or 3.11 for this reference environment"; assert sys.prefix != sys.base_prefix, "Activate a dedicated virtual environment"'
python -c 'import ssl; print("Python SSL:", ssl.OPENSSL_VERSION)'
# Driver 515.76 reported on ctl: use CUDA 11.x minor compatibility, then smoke-test.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu118}"
export PIP_NO_CACHE_DIR=1
python -m pip install 'pip==24.2' 'setuptools==75.1.0' 'wheel==0.44.0'
# Pin binary image/array dependencies before Torchvision resolves their versions.
python -m pip install --only-binary=:all: 'numpy==1.23.5' 'Pillow==9.5.0'
python -m pip install --only-binary=:all: -c "$SCRIPT_DIR/requirements.txt" 'torch==2.4.1' 'torchvision==0.19.1' --index-url "$TORCH_INDEX"
python -m pip install -r "$SCRIPT_DIR/requirements.txt"
python -m pip check
python -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda, "GPU available", torch.cuda.is_available()); assert torch.cuda.is_available(), "Check GPU and NVIDIA driver compatibility"'
