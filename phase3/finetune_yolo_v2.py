"""
Phase 3 (new scene) - Tier 2 fix, v2: fine-tune YOLOv8 on the auto-labeled
dataset from gen_yolo_dataset_v2.py (dual-node deployment: TX1/TX2 wide
views + ped_a/ped_b close-up nodes).

Same rationale as finetune_yolo.py (detection-only yolov8n, not -seg -- the
pipeline never reads res.masks) and the same mlflow-import workaround for
this environment's numpy/pandas ABI mismatch.

Usage:
    .venv\\Scripts\\python.exe phase3\\finetune_yolo_v2.py
"""
import sys
import types
from pathlib import Path

if "mlflow" not in sys.modules:
    sys.modules["mlflow"] = types.ModuleType("mlflow")

from ultralytics import YOLO

WORKSPACE = Path(r"C:\Users\AdminFix\Desktop\PFA")
PHASE3_DIR = WORKSPACE / "phase3"
DATA_YAML = PHASE3_DIR / "yolo_dataset_v2" / "data.yaml"

assert DATA_YAML.exists(), f"Dataset introuvable : {DATA_YAML} (lancer gen_yolo_dataset_v2.py d'abord)"

model = YOLO("yolov8n.pt")

results = model.train(
    data=str(DATA_YAML),
    epochs=40,
    imgsz=320,   # CPU-only torch in this venv -- see finetune_yolo.py's note on why 320/batch=8
    batch=8,
    workers=0,
    patience=10,
    project=str(PHASE3_DIR),
    name="yolo_finetune_v2",
    exist_ok=True,
    verbose=True,
)

best_weights = PHASE3_DIR / "yolo_finetune_v2" / "weights" / "best.pt"
print(f"\nEntrainement termine. Poids fine-tunes -> {best_weights}")
assert best_weights.exists(), "best.pt introuvable -- l'entrainement a-t-il echoue ?"

metrics = model.val(data=str(DATA_YAML))
print("\n=== Metriques de validation (fine-tune v2) ===")
print(f"mAP50    : {metrics.box.map50:.3f}")
print(f"mAP50-95 : {metrics.box.map:.3f}")
