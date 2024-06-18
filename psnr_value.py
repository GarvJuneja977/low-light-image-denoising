import os
import cv2
import numpy as np

def calculate_peak_signal_to_noise_ratio(ref_image_path, dist_image_path):
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between a reference and a distorted image."""
    # Load the reference and distorted images
    ref_img = cv2.imread(ref_image_path)
    dist_img = cv2.imread(dist_image_path)
    
    # Check if images loaded successfully
    if ref_img is None:
        print(f"Error: Could not load reference image from {ref_image_path}")
        return None
    if dist_img is None:
        print(f"Error: Could not load distorted image from {dist_image_path}")
        return None

    # Convert images to grayscale
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    dist_gray = cv2.cvtColor(dist_img, cv2.COLOR_BGR2GRAY)

    # Resize distorted image to match the dimensions of the reference image
    dist_gray = cv2.resize(dist_gray, (ref_gray.shape[1], ref_gray.shape[0]))

    # Compute MSE (Mean Squared Error)
    mse = np.mean((ref_gray - dist_gray) ** 2)

    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        psnr = 100  # PSNR is infinity if mse is zero
    else:
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    return psnr

def calculate_psnr_for_all_images(ref_folder_path, dist_folder_path):
    """Calculate PSNR for all image pairs in the specified folders."""
    ref_images = os.listdir(ref_folder_path)
    dist_images = os.listdir(dist_folder_path)

    # Assert number of images in both folders are the same
    assert len(ref_images) != len(dist_images), "Number of images in reference and distorted folders must be the same."

    psnr_values = []
    for ref_img_name, dist_img_name in zip(ref_images, dist_images):
        ref_img_path = os.path.join(ref_folder_path, ref_img_name)
        dist_img_path = os.path.join(dist_folder_path, dist_img_name)

        psnr_value = calculate_peak_signal_to_noise_ratio(ref_img_path, dist_img_path)
        # Append valid PSNR values to the list
        if psnr_value is not None:
            psnr_values.append(psnr_value)
            print(f'PSNR for {ref_img_name} and {dist_img_name}: {psnr_value:.2f} dB')

    return psnr_values

# Example usage:
ref_images_folder = '/content/drive/MyDrive/garvproject/high'
dist_images_folder = '/content/drive/MyDrive/garvproject/low'
psnr_values = calculate_psnr_for_all_images(ref_images_folder, dist_images_folder)

# Optionally, compute the average PSNR across all images
if psnr_values:
    average_psnr = np.mean(psnr_values)
    print(f'Average PSNR across all images: {average_psnr:.2f} dB')
else:
    print('No valid PSNR values calculated.')
