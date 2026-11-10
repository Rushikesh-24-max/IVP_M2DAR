# Frozen epoch-170 independent test results

Checkpoint SHA256: `f2a254dca49cd5156b344d3abff40e21061bb3fc4e605ba1728925d2d3db3115`. No training or checkpoint selection used this test set.

## Primary test: all 1,748 per-camera clips from 19 unseen drivers

| Metric | Result |
|---|---:|
| Accuracy | 84.84% |
| Macro-F1 | 84.87% |
| Macro-precision | 85.08% |
| Macro-recall / balanced accuracy | 85.04% |
| Top-5 accuracy | 97.71% |

Accuracy driver-cluster bootstrap 95% interval: 81.48%–87.81%.

Macro-F1 driver-cluster bootstrap 95% interval: 81.63%–87.76%.

## Per-camera primary test

| Camera | Clips | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| dashboard | 583 | 83.88% | 84.05% |
| rearview | 582 | 84.02% | 83.70% |
| right_side_window | 583 | 86.62% | 86.65% |

## Synchronization assessment

Original source videos checked: 114; source recording sessions: 38. Candidate aligned groups: 458; verified: 204; excluded or inconclusive: 254.

Fusion subset: **87.75% accuracy**, **88.67% macro-F1 over supported classes**, 204 verified groups from 19 drivers. Supported class IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].
All-16-class macro-F1 (zero for unsupported classes): 88.67%. Do not present the subset score as performance on all 1,748 clips.

The same synchronized subset before fusion has 612 individual camera views: 84.31% accuracy and 85.23% macro-F1. This is the matching population for assessing three-camera averaging.

| Metric | Three-camera fusion |
|---|---:|
| Macro-precision | 89.52% |
| Macro-recall / balanced accuracy | 89.06% |
| Top-5 accuracy | 98.04% |

Accuracy driver-cluster bootstrap 95% interval: 82.98%–91.62%.

Macro-F1 driver-cluster bootstrap 95% interval: 82.53%–91.57%.

## Interpretation

- A1 held-out-driver classification, not A2 ground-truth accuracy or temporal localization performance.
- Primary per-camera clips use each camera supplied action interval; previous validation used common aligned intervals, so clip selection differs.
- Synchronized fusion covers only informative groups passing source and post-crop motion checks; report coverage and missing classes.
- Synchronization is visually estimated; no hardware clock reference.
- Intervals quantify sampling uncertainty across the 19 supplied driver IDs within this dataset, not guaranteed real-world performance.

The epoch-170 checkpoint was selected using the earlier eight-driver A1 validation set. This separate test set had 19 unseen drivers. The dataset preparation, label mapping and synchronization policy were fixed without reading these model predictions. Keep this test set for final reporting; if future model changes are tuned to its outcomes, use a new untouched test set.
