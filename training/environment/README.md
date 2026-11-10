# GPU environment

The target is Linux x86_64 with Python 3.10 or 3.11 and one NVIDIA CUDA GPU. The code uses the vendored M2DAR package directly; do not install another `slowfast` package over it. No model environment or GPU training has been run on this Windows preprocessing machine. Python 3.11 is an additional candidate for the server's available `python-3.11.4` module; neither candidate has passed the server smoke test yet.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
bash environment/setup_linux.sh
```

The proposed pair is PyTorch 2.4.1 / torchvision 0.19.1 with CUDA 11.8 wheels. The default was changed after the user reported driver 515.76 on `ctl`; the earlier CUDA 12.1 default needs a newer driver. CUDA 11.8 is a candidate under NVIDIA's CUDA 11.x minor-version compatibility, which has feature/PTX limitations. The actual GPU smoke test is still required. The CUDA version displayed by `nvidia-smi` describes driver support, rather than the PyTorch wheel's bundled runtime or an installed toolkit. Newer GPU architectures may require a newer CUDA/PyTorch stack; adapt and rerun preflight instead of forcing these pins. `TORCH_INDEX` can override the official wheel index when appropriate.

On this server, first check scratch space, home quota, full GPU names, and availability. All four GPUs were occupied in the pasted snapshot, and `/data` was almost full. Loading `module load python-3.11.4` is a way to access a newer Python without changing the system Python. Use `python3.11 -m venv .venv`, activate it, then run the installer. Installation disables pip's download cache to reduce duplicate disk use. Reload the same Python module before reactivating the environment in a new SSH session. Run GPU tests and training only on an available assigned GPU; `CUDA_VISIBLE_DEVICES` maps the selected physical GPU to the scripts' logical GPU 0.

NumPy/Pillow are held to older versions for this 2023 codebase. FairScale provides activation checkpointing. Decord decodes video on CPU; NVIDIA video-decoding support is not required. The detection-only Detectron2 import has been made lazy; it is not required for MViT classification. TensorFlow, ONNX Runtime, bitsandbytes, and PyTorchVideo are not required by this configured path.

The installer checks SSL before downloading packages, installs the pinned NumPy/Pillow wheels first, and constrains the PyTorch installation to the requirements file. This avoids fetching newer NumPy/Pillow releases before the project's pins are applied. Binary-only selection in those steps avoids an unintended Pillow source build on glibc 2.17.

The reported site Python 3.11.4 failed to import SSL. The user instead created a Python 3.10 prefix with the available Anaconda module at `~/m2dar-training/python-ssl`, then a virtual environment at `~/m2dar-training/.venv-ssl`. Its HTTPS downloads succeeded. Reuse that working environment on this server; do not recreate it from the site Python without first checking `import ssl`.

The actual server environment must pass `preflight.py --gpu-smoke` before `train.py` starts. This checks real 448-pixel frames, model output, loss, gradients, one disposable AdamW step, and repeated validation sampling. A successful preflight records `pip freeze` and GPU details. The requirements file pins direct dependencies; this freeze records the resolved environment.

Decord's published Linux 0.6.0 wheel has an upstream metadata defect: its filename declares `py3-none`, while its internal WHEEL tag declares `cp36-cp36m`. This can make `pip check` reject a working installation on newer Python versions ([upstream issue 356](https://github.com/dmlc/decord/issues/356)). Run `python environment/check_decord.py` with the included small H.264 clip. Only after the native decoder successfully reads a real 16-frame batch will it correct that exact known tag inside the active virtual environment and update its RECORD hash. Originals are backed up in `checks/`; native binaries are unchanged. The full dataset and GPU smoke test are still required. If the native import/decode fails or the metadata differs from the known defect, no repair is made.

Start with batch 3 and FP32 to follow the released configuration. If memory is insufficient, generate a separate bundle with `--batch-size 1 --amp` and record it as a different experiment; learning rate is not silently scaled. Do not reduce resolution without also auditing positional embeddings. Single-GPU execution is intentional for the first baseline; multi-GPU/DDP adaptation is a later, separately validated change.

Primary references checked while preparing this package:

- [M2DAR pinned source](https://github.com/PurdueDigitalTwin/M2DAR/tree/8fc6a3f76fc47207c4fa072dff7259f0debb5472)
- [PyTorch official version pairs and wheel indexes](https://pytorch.org/get-started/previous-versions/#v241)
- [Decord 0.6.0 distributions](https://pypi.org/project/decord/0.6.0/)
- [FairScale 0.4.13](https://pypi.org/project/fairscale/0.4.13/)
- [PyAV 12.3.0](https://pypi.org/project/av/12.3.0/)
- [NVIDIA CUDA minor-version compatibility and limitations](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [NumPy 1.23.5 supports Python 3.8–3.11](https://github.com/numpy/numpy/releases/tag/v1.23.5)
