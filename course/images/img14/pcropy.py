#!/usr/bin/env python3
"""
pcrop.py

Automatically crop white margins from an image.

Usage:
    python pcrop.py image.png

The cropped image is saved as:
    image_crop.png
"""

import sys
from PIL import Image
import numpy as np
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python pcrop.py image.png")
    sys.exit(1)

img_path = Path(sys.argv[1])
out_path = img_path.with_stem(img_path.stem + "_crop")

# Load image
img = Image.open(img_path).convert("RGB")
arr = np.asarray(img)

# Detect non-white pixels
mask = np.any(arr < 250, axis=2)

coords = np.argwhere(mask)
y0, x0 = coords.min(axis=0)
y1, x1 = coords.max(axis=0) + 1

# Crop and save
cropped = img.crop((x0, y0, x1, y1))
cropped.save(out_path)

print(f"✅ Cropped image saved as: {out_path}")

