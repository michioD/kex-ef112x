#!/bin/bash

set -e  # Exit immediately if a command fails

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
DATA_DIR="$ROOT_DIR/datasets/coco_images"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

mkdir -p "$DATA_DIR"

# Download COCO validation images
curl --fail --location --retry 3 --output "$WORK_DIR/val2017.zip" https://images.cocodataset.org/zips/val2017.zip

# Unzip into target directory
unzip -q "$WORK_DIR/val2017.zip" -d "$DATA_DIR"

# Download COCO train/val 2017 annotations
curl --fail --location --retry 3 --output "$WORK_DIR/annotations_trainval2017.zip" https://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip annotations into target directory
# Note: The zip file inherently contains an 'annotations' directory at its root.
# Extracting it to datasets/coco_images will result in datasets/coco_images/annotations/instances_val2017.json
unzip -q "$WORK_DIR/annotations_trainval2017.zip" -d "$DATA_DIR"

echo "COCO val2017 dataset and annotations downloaded and extracted successfully."
