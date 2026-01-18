#!/bin/bash

# List of files to process
files=("billing_gpt_4_turbo_in.png" "billing_gpt_4_turbo_out.png")

for file in "${files[@]}"; do
    # Get image width and height using identify
    dimensions=$(identify -format "%w %h" "$file")
    width=$(echo $dimensions | cut -d' ' -f1)
    height=$(echo $dimensions | cut -d' ' -f2)

    # Calculate half width
    half_width=$((width / 2))

    # Output filename
    output="${file%.png}2.png"

    echo "Cropping $file to $output ($half_width x $height)..."
    convert "$file" -crop "${half_width}x${height}+0+0" "$output"
done

echo "✅ Cropping done."

