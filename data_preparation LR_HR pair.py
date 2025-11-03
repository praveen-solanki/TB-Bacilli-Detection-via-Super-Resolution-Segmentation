import os
import cv2
import numpy as np
from tqdm import tqdm

def create_lr_images(hr_folder, lr_folder, scale_factor=4):
    """
    Creates low-resolution (LR) images from high-resolution (HR) images.

    Args:
        hr_folder (str): Path to the folder containing HR images.
        lr_folder (str): Path to the folder where LR images will be saved.
        scale_factor (int): The factor by which to downscale the HR images.
    """
    if not os.path.exists(lr_folder):
        os.makedirs(lr_folder)

    hr_images = [f for f in os.listdir(hr_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
    
    if not hr_images:
        print(f"Error: No images found in the HR folder: {hr_folder}")
        return

    print(f"Found {len(hr_images)} HR images. Starting LR image generation...")

    for image_name in tqdm(hr_images, desc="Generating LR Images"):
        hr_image_path = os.path.join(hr_folder, image_name)
        
        # Read the HR image
        hr_image = cv2.imread(hr_image_path)
        if hr_image is None:
            print(f"Warning: Could not read image {image_name}. Skipping.")
            continue
            
        hr_height, hr_width, _ = hr_image.shape
        
        # Ensure HR dimensions are divisible by the scale factor
        hr_height = (hr_height // scale_factor) * scale_factor
        hr_width = (hr_width // scale_factor) * scale_factor
        hr_image = hr_image[0:hr_height, 0:hr_width]

        # Calculate LR dimensions
        lr_height = hr_height // scale_factor
        lr_width = hr_width // scale_factor

        # Downscale the image using bicubic interpolation
        lr_image = cv2.resize(hr_image, (lr_width, lr_height), interpolation=cv2.INTER_CUBIC)

        # Save the LR image
        lr_image_path = os.path.join(lr_folder, image_name)
        cv2.imwrite(lr_image_path, lr_image)
        
    print("LR image generation complete.")


if __name__ == '__main__':
    # --- Configuration ---
    SOURCE_HR_DIR = 'C:\\Users\\praveen solanki\\Desktop\\DDS3'
    DATASET_DIR = 'TB_dataset'
    TARGET_HR_DIR = os.path.join(DATASET_DIR, 'HR')
    TARGET_LR_DIR = os.path.join(DATASET_DIR, 'LR')
    DOWNSCALE_FACTOR = 8

    # --- Step 1: Check Source Directory ---
    if not os.path.exists(SOURCE_HR_DIR):
        # os.makedirs(SOURCE_HR_DIR)
        # print(f"Created source directory: '{SOURCE_HR_DIR}'")
        # print("------------------------------------------------------------------")
        # print("ACTION REQUIRED: Please add your high-resolution TB bacilli images")
        # print(f"to the '{SOURCE_HR_DIR}' directory and run this script again.")
        # print("------------------------------------------------------------------")
        exit()

    source_files = [f for f in os.listdir(SOURCE_HR_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
    if not source_files:
        print(f"Error: The source directory '{SOURCE_HR_DIR}' is empty.")
        print("------------------------------------------------------------------")
        print("ACTION REQUIRED: Please add your high-resolution TB bacilli images")
        print(f"to the '{SOURCE_HR_DIR}' directory and run this script again.")
        print("------------------------------------------------------------------")
        exit()
        
    print(f"Found {len(source_files)} images in '{SOURCE_HR_DIR}'.")

    # --- Step 2: Setup Dataset Directories ---
    os.makedirs(TARGET_HR_DIR, exist_ok=True)
    os.makedirs(TARGET_LR_DIR, exist_ok=True)

    # --- Step 3: Create Paired Dataset ---
    print("\nCopying HR images to the new dataset folder...")
    for file_name in tqdm(source_files, desc="Copying HR images"):
        source_path = os.path.join(SOURCE_HR_DIR, file_name)
        target_path = os.path.join(TARGET_HR_DIR, file_name)
        import shutil
        shutil.copy(source_path, target_path)
    
    print("\nStarting the generation of Low-Resolution (LR) images...")
    create_lr_images(TARGET_HR_DIR, TARGET_LR_DIR, DOWNSCALE_FACTOR)
    
    print(f"\nDataset preparation finished.")
    print(f"Your paired dataset is ready in the '{DATASET_DIR}' folder.")
    print(f"  - High-Resolution images are in: '{TARGET_HR_DIR}'")
    print(f"  - Low-Resolution images are in:  '{TARGET_LR_DIR}'")

