"""Fine-tuning YOLO v3 (car, pedestrian, bus) sur le jeu de donnees de yolo_v3_donnees.py.

Point de depart : le modele v2 de la phase 3 (deja adapte aux rendus Mitsuba) ; la tete est
reconstruite pour 3 classes. Entrainement et inference en 960 px (pietons lointains : ~30 px de haut
dans une image 1280 x 960, trop petits une fois reduits a 640 ou 320 px).

Usage : .venv\\Scripts\\python.exe "Digital Twin assisted beamforming design\\code\\yolo_v3_entrainement.py"
            [--resume]   reprendre apres une interruption
            [--val]      seulement valider les meilleurs poids
Sortie : yolo_v3/train/weights/best.pt

Sous Windows, les sous-processus de chargement des donnees re-executent ce fichier : tout le code est
donc sous `if __name__ == "__main__"`, et la validation utilise workers=0.
"""
import sys
import types

if "mlflow" not in sys.modules:
    sys.modules["mlflow"] = types.ModuleType("mlflow")


def main():
    import torch
    from ultralytics import YOLO

    import dt_common as dc

    data = dc.ROOT / "yolo_v3" / "dataset" / "data.yaml"
    base = dc.PHASE3_DIR / "yolo_finetune_v2" / "weights" / "best.pt"
    last = dc.ROOT / "yolo_v3" / "train" / "weights" / "last.pt"
    best = dc.ROOT / "yolo_v3" / "train" / "weights" / "best.pt"
    imgsz = 960

    assert data.exists(), f"{data} introuvable (lancer yolo_v3_donnees.py d'abord)"
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}, appareil : {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    if "--val" not in sys.argv:
        if "--resume" in sys.argv and last.exists():
            print(f"reprise depuis {last}")
            YOLO(str(last)).train(resume=True, workers=0)
        else:
            YOLO(str(base)).train(data=str(data), epochs=60, patience=15, imgsz=imgsz,
                                  batch=16 if device == 0 else 8, workers=0, device=device,
                                  project=str(dc.ROOT / "yolo_v3"), name="train", exist_ok=True,
                                  degrees=0.0, fliplr=0.5, scale=0.5, mosaic=1.0, close_mosaic=10, seed=0,
                                  verbose=True)
    m = YOLO(str(best)).val(data=str(data), imgsz=imgsz, device=device, workers=0,
                            project=str(dc.ROOT / "yolo_v3"), name="val", exist_ok=True)
    print(f"\nmeilleurs poids : {best}")
    print(f"validation : mAP50 {m.box.map50:.3f}, mAP50-95 {m.box.map:.3f}")
    for i, c in m.names.items():
        print(f"  {c:11s} AP50 {m.box.ap50[i]:.3f}  AP50-95 {m.box.maps[i]:.3f}")


if __name__ == "__main__":
    main()
