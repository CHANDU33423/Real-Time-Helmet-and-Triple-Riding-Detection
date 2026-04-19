"""
dataset_manager.py — Utility to prepare and verify YOLO dataset for training.
"""

import os

DATASET_DIR = "dataset"
YAML_FILE = os.path.join(DATASET_DIR, "data.yaml")


def check_structure():
    print(f"\n[Dataset] Checking structure in '{DATASET_DIR}'...")

    required = ["images/train", "images/val", "labels/train", "labels/val"]

    missing = []
    for rel_path in required:
        full_path = os.path.join(DATASET_DIR, rel_path)
        if not os.path.exists(full_path):
            missing.append(rel_path)

    if not missing:
        print("[Dataset] Structure is VALID ✅")
    else:
        print(f"[Dataset] Missing directories: {missing} ❌")

    if os.path.exists(YAML_FILE):
        print(f"[Dataset] data.yaml found at {YAML_FILE} ✅")
    else:
        print(f"[Dataset] data.yaml MISSING ❌")


def setup_demo_dataset():
    """Placeholder for downloading a demo dataset from a public source."""
    # Example: Roboflow public URL
    # URL = "https://public.roboflow.com/ds/..."
    print("[Dataset] Demo dataset download would happen here.")
    print(
        "[Dataset] For now, please manually place your YOLO images/labels in the 'dataset' folder."
    )


if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    check_structure()

    print("\nNext Steps:")
    print("1. Place your training images in 'dataset/images/train'")
    print("2. Place your validation images in 'dataset/images/val'")
    print("3. Place your YOLO label txt files in 'dataset/labels/train' & 'val'")
    print("4. Run 'python train.py' to start training your custom model.")
