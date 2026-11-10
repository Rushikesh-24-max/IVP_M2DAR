# Restore inference on another GPU machine

The restore kit contains the exact epoch-170 model, its inference configuration, a small provenance manifest, and the training and evaluation results. A separate Linux machine with a compatible NVIDIA GPU can run the app with these files.

1. Clone this repository. Download both parts from the [project Drive folder](https://drive.google.com/drive/folders/1bsEZWij3E3UBur9wAQjTdPJvRDZD1UaB).
   - [Part 1, 70 MiB](https://drive.google.com/file/d/1Liq3Hixv_xuCe8q8LP3y2rd4cnUmoIkW/view)
   - [Part 2, 56 MiB](https://drive.google.com/file/d/1rEpbkGyLDmcpKVRJKptb9qjka1G-Qy3E/view)
2. From the repository root, run `python tools/assemble_restore_kit.py /path/to/downloads`. The script uses the checksums in this repository to verify both parts and the assembled ZIP.
3. Extract the verified ZIP into a directory called `epoch170` outside the checkout. On Linux, run `sha256sum -c SHA256SUMS.txt` from that directory to verify its contents.
4. Set up the Linux environment using the [dependency instructions](../training/environment/README.md). The institute run subsequently completed GPU preflight, training, and inference. A different machine still needs its own readiness check.

From the repository root on the GPU machine, set `kit` to the absolute path of your extracted kit and run the worker's readiness check.

```bash
repo="$PWD"
kit=/absolute/path/to/epoch170
python app/backend/gpu_worker.py --check \
  --training-root "$repo/training" \
  --bundle "$kit/inference_bundle" \
  --checkpoint "$kit/checkpoint/best.pyth" \
  --readiness-out "$repo/app/.runtime/readiness.json"
```

After the readiness check passes, set a private `M2DAR_API_TOKEN` of at least 24 characters in the server shell and start the app.

```bash
cd "$repo/app"
python backend/server.py --host 127.0.0.1 --port 8765 \
  --training-root "$repo/training" \
  --bundle "$kit/inference_bundle" \
  --checkpoint "$kit/checkpoint/best.pyth"
```

If the GPU server is remote, open the tunnel from the laptop that will run the browser. Use your authorized SSH username and the reachable GPU hostname.

```bash
ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 your_user@gpu-host
```

Open `http://localhost:8765`, enter the API token in connection settings, and select **Full video** for a timeline of predictions. The inference bundle is sufficient for new videos. Resuming training requires a model-plus-optimizer checkpoint and the original prepared split. Repeating the reported evaluation requires authorized access to the challenge data and locally prepared evaluation manifests.
