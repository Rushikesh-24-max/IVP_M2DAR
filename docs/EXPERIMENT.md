# Experiment and results

The experiment adapted the released M2DAR MViTv2-S configuration for 16 driver action classes. It used a generic Kinetics-700 initializer, 16-frame inputs sampled at stride 4, 448-pixel crops, AdamW with a base learning rate of `5e-6`, and 200 epochs. The completed run used batch size 1, mixed precision, seed 42, and a single GPU.

| Population | Drivers | Samples | Use |
|---|---:|---:|---|
| A1 training | 32 | 2,325 camera clips | Model fitting |
| A1 validation | 8 | 456 camera clips | Select epoch 170 by macro-F1 |
| A1_5 and A1_6 independent test | 19 | 1,748 camera clips | Primary held-out evaluation |
| Verified synchronized test subset | 19 | 204 three-camera groups | Secondary three-camera averaging |

| Evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| Validation, individual clips | 90.35% | 90.46% |
| Validation, three-camera groups | 94.08% | 94.20% |
| Independent test, all individual clips | 84.84% | 84.87% |
| Same synchronized test subset, 612 individual views | 84.31% | 85.23% |
| Same synchronized test subset, 204 averaged groups | 87.75% | 88.67% |

The independent test used the already selected checkpoint. Each single-camera clip received a deterministic centre-window prediction. The three-camera result averages the three probability vectors within each verified group. The subset was selected using visual synchronization evidence and corresponding labels. The comparison between 612 views and 204 averaged groups uses the same underlying synchronized clips. A2 yielded inference outputs only because local ground-truth labels were unavailable.

## Experiment records

- [`training_history.csv`](../results/training_history.csv) records loss, learning rate, and top-1/top-5 error for all 200 epochs. Error columns ending in `_percent` use the 0–100 scale.
- [`validation_history.csv`](../results/validation_history.csv) records the 20 validation evaluations, including the maximum macro-F1 at epoch 170. Accuracy and F1 columns use the 0–1 scale.
- [`results/README.md`](../results/README.md) reports class and camera results, uncertainty intervals, and the synchronization criteria.
- [`resolved.public.yaml`](../results/resolved.public.yaml) records the training settings with portable path placeholders. The original resolved config and source hashes are recorded in [`run_provenance.json`](../results/run_provenance.json).
- [Restore guide](RESTORE.md) shows how to load the exact model on another GPU host.
