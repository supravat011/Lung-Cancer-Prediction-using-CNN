"""
Example script demonstrating how to use the prediction script
This will test the model with sample images from the dataset
"""

import os
import subprocess
import sys

# Sample images from each class
SAMPLE_IMAGES = [
    "dataset/test/normal/000119.png",
    "dataset/test/adenocarcinoma/000130 (6).png",
    "dataset/test/largecell/000033 (2).png",
    "dataset/test/squamouscell/000015 (7).png"
]

def main():
    print("="*60)
    print("  Testing Lung Cancer Prediction Model")
    print("="*60)
    print()
    
    # Check if predict.py exists
    if not os.path.exists("predict.py"):
        print("Error: predict.py not found!")
        print("Please run this script from the project directory.")
        sys.exit(1)
    
    # Find available test images
    available_images = []
    for img_path in SAMPLE_IMAGES:
        if os.path.exists(img_path):
            available_images.append(img_path)
    
    if not available_images:
        print("No sample images found in the dataset folder.")
        print("Please ensure the dataset folder contains test images.")
        sys.exit(1)
    
    print(f"Found {len(available_images)} sample image(s) to test.\n")
    
    # Run prediction on all available images
    print("Running predictions...\n")
    cmd = ["python", "predict.py"] + available_images
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
