import os
import cv2
import numpy as np

def psnr(img1, img2):
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if images are identical
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def calculate_psnr_between_folders(input_folder, output_folder):
    input_images = []
    output_images = []
    psnr_values = []

    # List all files in input and output folders
    input_files = sorted(os.listdir(input_folder))
    output_files = sorted(os.listdir(output_folder))

    # Iterate through each pair of images
    for input_file, output_file in zip(input_files, output_files):
        if input_file.endswith('.jpg') or input_file.endswith('.png'):
            input_img_path = os.path.join(input_folder, input_file)
            output_img_path = os.path.join(output_folder, output_file)

            # Read images
            img_input = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
            img_output = cv2.imread(output_img_path, cv2.IMREAD_GRAYSCALE)

            if img_input is None:
                print(f"Warning: Input image not found or could not be read: {input_img_path}")
                continue
            if img_output is None:
                print(f"Warning: Output image not found or could not be read: {output_img_path}")
                continue

            # Calculate PSNR
            psnr_value = psnr(img_input, img_output)
            psnr_values.append(psnr_value)

            # Print PSNR value for the current pair
            print(f"PSNR between {input_file} and {output_file}: {psnr_value:.2f} dB")

    # Calculate average PSNR
    average_psnr = np.mean(psnr_values)
    print(f"\nAverage PSNR across all pairs: {average_psnr:.2f} dB")

# Example usage:
if __name__ == '__main__':
    input_folder = '/content/drive/MyDrive/garvproject/low'
    output_folder = '/content/drive/MyDrive/garvproject/high'

    calculate_psnr_between_folders(input_folder, output_folder)
