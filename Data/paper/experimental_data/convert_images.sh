#!/bin/bash

# Navigate to the directory containing the PNG files
cd segmented_images/

# Loop through each PNG file in the directory
for file in *.png; do
    # Check if the file is a regular file (not a directory)
    if [ -f "$file" ]; then
        # Extract the filename without the extension
        filename=$(basename "$file" .png)
        
        # Convert the PNG file to TIFF format
        convert "$file" "${filename}.tiff"
        
        # Optional: Delete the original PNG file if needed
        # rm "$file"
    fi
done
