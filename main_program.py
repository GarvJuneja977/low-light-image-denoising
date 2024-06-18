import cv2
import numpy as np
import os
from os import listdir
from matplotlib import pyplot as plt
from bm3d import bm3d

from required_functions import refine_illumination_map, apply_gamma_correction, perform_bm3d_denoising


input_folder = '/content/drive/MyDrive/garvproject/low'
output_folder = '/content/drive/MyDrive/garvproject/high'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

weight_strategy = 4
gamma_correction = 0.5
sigma = 0.05

def is_image(filename):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return filename.lower().endswith(valid_extensions)

image_files = [file for file in listdir(input_folder) if is_image(file)]

for filename in image_files:
    filepath = os.path.join(input_folder, filename)
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)

    # Convert image to RGB and normalize
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    # Calculate illumination map
    illumination_map = np.max(image_rgb, axis=-1)

    # Update illumination map
    updated_illum_map = refine_illumination_map(illumination_map, weight_strategy)

    # Apply gamma correction
    gamma_corrected_illum_map = apply_gamma_correction(np.abs(updated_illum_map), gamma_correction)
    gamma_corrected_illum_map = gamma_corrected_illum_map[..., np.newaxis]

    # Correct image illumination
    corrected_image = image_rgb / gamma_corrected_illum_map

    # Clip image values and convert to float32
    corrected_image = np.clip(corrected_image, 0, 1).astype(np.float32)

    # Denoise image using BM3D
    denoised_image = perform_bm3d_denoising(corrected_image, gamma_corrected_illum_map, sigma)

    # Plot original and denoised images
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image)
    plt.title('Predicted Image')

    # Save denoised image
    denoised_image_bgr = cv2.cvtColor(denoised_image * 255, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_folder, 'processed_' + filename), denoised_image_bgr)

    plt.show()
