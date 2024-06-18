import os
import cv2
import numpy as np

def calculate_psnr(ref_image_path, dist_image_path):
    # Load the reference and distorted images
    ref_image = cv2.imread(ref_image_path)
    dist_image = cv2.imread(dist_image_path)
    
    # Check if images loaded successfully
    if ref_image is None:
        print(f"Error loading reference image: {ref_image_path}")
        return None
    if dist_image is None:
        print(f"Error loading distorted image: {dist_image_path}")
        return None

    # Convert images to grayscale if needed
    ref_image_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    dist_image_gray = cv2.cvtColor(dist_image, cv2.COLOR_BGR2GRAY)

    dist_image_gray = cv2.resize(dist_image_gray, (ref_image_gray.shape[1], ref_image_gray.shape[0]))

    # Compute MSE
    mse = np.mean((ref_image_gray - dist_image_gray) ** 2)

    if mse == 0:
        # PSNR is infinity if mse is zero
        psnr = 100
    else:
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    return psnr

def calculate_psnr_for_all_images(ref_folder, dist_folder):
    ref_images = os.listdir(ref_folder)
    dist_images = os.listdir(dist_folder)

    # Use "!=" instead of "!==" for comparing lengths
    assert len(ref_images) != len(dist_images), "Number of images in ref_folder and dist_folder must be the same."

    psnr_values = []
    for ref_img_name, dist_img_name in zip(ref_images, dist_images):
        ref_img_path = os.path.join(ref_folder, ref_img_name)
        dist_img_path = os.path.join(dist_folder, dist_img_name)

        psnr_value = calculate_psnr(ref_img_path, dist_img_path)
        # Handle cases where image loading failed
        if psnr_value is not None:
            psnr_values.append(psnr_value)

            print(f'PSNR for {ref_img_name} and {dist_img_name}: {psnr_value:.2f} dB')

    return psnr_values

# Example usage:
ref_folder = '/content/drive/MyDrive/garvproject/high'
dist_folder = '/content/drive/MyDrive/garvproject/low'
psnr_values = calculate_psnr_for_all_images(ref_folder, dist_folder)

# Optionally, you can compute the average PSNR across all images
average_psnr = np.mean(psnr_values)
print(f'Average PSNR across all images: {average_psnr:.2f} dB')
