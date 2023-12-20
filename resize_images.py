'''
This file is for preparing the images for preprocessing which will later be used in the unet model
'''

import glob
import os
from PIL import Image

# Define the target size for resizing
target_size = (256, 256)

# Load the Input Images
initial_img_dir = os.path.join(os.getcwd(), 'data/initial_imgs')
img_dataset = sorted(glob.glob(os.path.join(initial_img_dir, "*/*.jpg")))

#  Create a directory to save resized images
output_folder = os.path.join(os.getcwd(), 'data/resized_imgs')
os.makedirs(output_folder, exist_ok=True)

# Loop through each image and perform resizing
for img_path in img_dataset:
    # Load the image
    img = Image.open(img_path)

    # Resize the image
    resized_img = img.resize(target_size)

    # Extract the relative path from the initial_img_dir
    relative_path = os.path.relpath(img_path, initial_img_dir)

    # Create the corresponding path in the resized_img_dir
    resized_img_path = os.path.join(output_folder, relative_path)

    # Create the directory structure if it doesn't exist
    os.makedirs(os.path.dirname(resized_img_path), exist_ok=True)

    # Save the resized image in the new directory
    resized_img.save(resized_img_path)

    print(f"Resized and saved: {resized_img_path}")
