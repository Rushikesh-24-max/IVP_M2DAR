"""Metrics and best inference checkpoint hook used by the recorded source patch."""
from pathlib import Path
import json

from common import read_json, write_json
from metrics import classification_metrics


def record_validation(model, meter, epoch, cfg, writer=None):
    import torch
    probabilities = torch.cat(meter.all_preds).detach().cpu().tolist()
    labels = torch.cat(meter.all_labels).detach().cpu().tolist()
    result = classification_metrics(labels, probabilities, cfg.MODEL.NUM_CLASSES)
    result.update(epoch=epoch + 1, protocol="one center temporal window and center spatial crop per view clip")
    run = Path(cfg.OUTPUT_DIR)
    metadata = read_json(run / "run.json")
    if result["samples"] != metadata["validation_clips"]:
        raise ValueError("Validation did not cover every clip")
    write_json(run / "metrics" / f"epoch_{epoch+1:04d}.json", result)
    with (run / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, allow_nan=False) + "\n")
    if writer is not None:
        writer.add_scalars({"Val/Macro_F1": result["macro_f1"], "Val/Balanced_Accuracy": result["balanced_accuracy"]},
                           global_step=epoch + 1)
    best_file = run / "best_metrics.json"
    previous = read_json(best_file)["macro_f1"] if best_file.exists() else -1.
    if result["macro_f1"] > previous:
        checkpoint = dict(model_state=model.state_dict(), epoch=epoch, cfg=cfg.dump(),
                          purpose="inference_only", dataset_fingerprint=metadata["dataset_fingerprint"],
                          run_id=metadata["run_id"])
        temp = run / "best.pyth.tmp"
        torch.save(checkpoint, temp)
        temp.replace(run / "best.pyth")
        write_json(best_file, result)
    print(f"Validation epoch {epoch+1}: accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}")
    from checkpoint_storage import retain_recent_checkpoints
    retain_recent_checkpoints(run, epoch + 1)
