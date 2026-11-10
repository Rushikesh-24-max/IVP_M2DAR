# Local app

From this directory, run `python backend/server.py --demo-only` and open `http://localhost:8765`. You can preview one local video and read the aggregate results. Saved challenge recordings are supplied separately only when authorized.

For GPU inference, use the tested Linux environment described in `../training/environment/README.md`. Provide the prepared training bundle, original epoch-170 checkpoint and its neighboring `run.json` from the authorized experiment archive. Set `M2DAR_API_TOKEN` privately and pass `--training-root`, `--bundle`, and `--checkpoint` to `python backend/server.py`. Select the GPU with `CUDA_VISIBLE_DEVICES` as appropriate for your host. The backend verifies the frozen checkpoint and source lock.

Forward the service from an authorized client with `ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 your_user@gpu-host`. Open the forwarded address, enter your authorized API token in connection settings and select Full video to enable time-varying scores. Completed jobs can be reopened using Inspect timeline.

Run `node tests/slider.test.cjs` for the timeline regression checks.
