"""
train.py — Custom YOLOv8 training script
Fine-tune YOLOv8 on your helmet/rider dataset.

Dataset structure expected (YOLO format):
    dataset/
        images/
            train/   *.jpg
            val/     *.jpg
        labels/
            train/   *.txt
            val/     *.txt
        data.yaml

data.yaml example:
    path: ./dataset
    train: images/train
    val:   images/val
    nc: 4
    names: ['motorcycle', 'rider', 'helmet', 'no_helmet']

Usage:
    python train.py
    python train.py --model yolov8s.pt --epochs 150 --batch 32
"""

import argparse
import os

from ultralytics import YOLO


def train(
    model_variant: str,
    data_yaml: str,
    epochs: int,
    imgsz: int,
    batch: int,
    lr0: float,
    output_name: str,
):

    print(f"\n{'='*60}")
    print(f"  YOLOv8 Custom Training -- Helmet Detection")
    print(f"{'='*60}")
    print(f"  Base model:   {model_variant}")
    print(f"  Dataset:      {data_yaml}")
    print(f"  Epochs:       {epochs}")
    # BUG FIX #15: replaced Unicode multiplication 'x' (crashes cp1252)
    print(f"  Image size:   {imgsz}x{imgsz}")
    print(f"  Batch size:   {batch}")
    print(f"  Learning rate:{lr0}")
    print(f"{'='*60}\n")

    if not os.path.exists(data_yaml):
        print(f"[ERROR] data.yaml not found at: {data_yaml}")
        print("  Please create your dataset following YOLO label format.")
        return

    model = YOLO(model_variant)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        optimizer="Adam",
        augment=True,
        flipud=0.0,
        fliplr=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,
        name=output_name,
        project="runs/train",
        exist_ok=True,
        patience=30,
        save=True,
        plots=True,
        device=0 if _has_gpu() else "cpu",
    )

    # BUG FIX #16: replaced emoji in print (crashes cp1252 Windows terminal)
    print("\n[Training] Training complete!")
    print(f"[Training] Best weights saved at: runs/train/{output_name}/weights/best.pt")
    print(f"[Training] Copy best.pt to:       models/best.pt  to use it in main.py")


def _has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on helmet dataset")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base YOLOv8 variant (yolov8n/s/m/l/x.pt) default: yolov8n.pt",
    )
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--name", default="helmet_detection_v1")
    args = parser.parse_args()

    train(
        args.model, args.data, args.epochs, args.imgsz, args.batch, args.lr, args.name
    )
