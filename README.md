# IVP M2DAR

Academic research implementation of the released M2DAR MViTv2-S configuration. We trained on 32 A1 drivers for 200 epochs and selected epoch 170 using a separate eight-driver validation set. A1_5 and A1_6 supplied 1,748 held-out camera clips from 19 drivers. The independent single-camera test reached 84.84% accuracy and 84.87% macro-F1. The 204 verified synchronized triplets reached 87.75% accuracy and 88.67% macro-F1 on that subset.

- `training/` contains the frozen training and inference code, upstream source lock, and environment setup.
- `preprocessing/` contains the clip and synchronization preparation scripts. These research scripts use workspace-specific path defaults. Set them to your local dataset and output roots before running.
- `evaluation/` contains the test and synchronization evaluators. See `results/README.md` for the protocol and limits. The original per-recording annotation and membership files must be prepared locally from authorized dataset access.
- `app/` contains the website and the single-video GPU backend. See `app/README.md` for setup.
- `results/` contains aggregate results and training provenance. `resolved.public.yaml` substitutes portable paths for the original server paths. The original config digest is recorded separately.

Obtain the research dataset directly from the [AI City Challenge Track 3 download](https://www.aicitychallenge.org/2024-track3-download/) and follow its [dataset agreement](https://www.aicitychallenge.org/wp-content/uploads/2024/01/DATASET_LICENSE_AGREEMENT_2024_Track3.pdf). Models trained on the dataset are limited to noncommercial research use under that agreement. Obtain the frozen checkpoint from the project's authorized team archive and verify SHA256 `f2a254dca49cd5156b344d3abff40e21061bb3fc4e605ba1728925d2d3db3115`.

The vendored M2DAR code retains Purdue Digital Twin Lab's [MIT notice](training/vendor/M2DAR/LICENSE). The original source is [PurdueDigitalTwin/M2DAR](https://github.com/PurdueDigitalTwin/M2DAR).

The public snapshot omits the upstream documentation figures. The source check still validates the frozen runtime files and checks the figure hashes if those files are present.

[Download the aggregate results and sanitized training configuration](https://drive.google.com/file/d/17DAOs2VW3KfsGy1VNghsQ2UOdH-R7fds/view).
