#!/usr/bin/env python3
"""
Move images referenced in chapters/chNN/chNN.tex
from images/old/ to images/imgNN/.
"""

import sys
import re
from pathlib import Path
import shutil

# ------------------------------------------------------------
# Usage
# ------------------------------------------------------------
if len(sys.argv) != 2:
    print("Usage: python mv_images.py chNN")
    sys.exit(1)

chapter = sys.argv[1]

# ------------------------------------------------------------
# Validate chapter name
# ------------------------------------------------------------
m = re.fullmatch(r"ch(\d+)", chapter)
if not m:
    print("Error: argument must be of the form chNN")
    sys.exit(1)

NN = m.group(1)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
base = Path(__file__).resolve().parent
tex_file = base / chapter / f"{chapter}.tex"

#old_dir = base.parent / "images" / "old"
old_dir = Path("/Users/rpotthas/all/e-ai_python_ml_tutorial/images")
new_dir = base.parent / "images" / f"img{NN}"
new_dir.mkdir(parents=True, exist_ok=True)

if not tex_file.exists():
    print(f"Error: {tex_file} not found")
    sys.exit(1)

# ------------------------------------------------------------
# Extract images from LaTeX
# ------------------------------------------------------------
pattern = re.compile(
    r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"
)

content = tex_file.read_text()
matches = pattern.findall(content)

if not matches:
    print("No images found.")
    sys.exit(0)

# ------------------------------------------------------------
# Move images
# ------------------------------------------------------------
moved = 0

for img in matches:
    img_name = Path(img).name
    src = old_dir / img_name
    dst = new_dir / img_name

    if not src.exists():
        print(f"[WARN] Missing: {src}")
        continue

    print(f"Moving {src} -> {dst}")
    shutil.move(src, dst)
    moved += 1

print(f"\nDone. Moved {moved} image(s) to {new_dir}")

