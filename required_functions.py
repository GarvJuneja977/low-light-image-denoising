import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from bm3d import bm3d
from typing import Union

def is_image_file(file_name: str) -> bool:
    """
    Checks if a file name ends with a recognized image format ('bmp', 'jpg', 'png', 'tif').

    Returns True if the file name ends with one of these formats, and False otherwise.
    """
    is_image = file_name[-3:] in ['bmp', 'jpg', 'png', 'tif']
    return is_image

def apply_gamma_correction(illumination_map: np.ndarray, gamma_value: Union[int, float]) -> np.ndarray:
    """
    Applies gamma correction to the input illumination map using the specified gamma coefficient.

    Args:
    - illumination_map (np.ndarray): Input illumination map array of shape (M, N).
    - gamma_value (Union[int, float]): Gamma coefficient used for correction.

    Returns:
    - np.ndarray: Shape-(M, N) array of the corrected illumination map.
    """
    corrected_map = illumination_map ** gamma_value
    return corrected_map


def calculate_loe_metric(ref_image: np.ndarray, comp_image: np.ndarray) -> float:
    """
    Calculates the Lightness Order Error (LOE) metric by comparing pixel intensities
    between a comparison image and its reference counterpart.

    Args:
    - ref_image (np.ndarray): Reference image array of shape (M, N).
    - comp_image (np.ndarray): Comparison (refined) image array of shape (M, N).

    Returns:
    - float: Calculated value of the LOE metric.
    """
    v_dim, h_dim = ref_image.shape
    total_pixels = ref_image.size
    loe_loss = 0

    # Vectorized calculation
    for v in range(v_dim):
        for h in range(h_dim):
            ref_condition = ref_image <= ref_image[v, h]
            comp_condition = comp_image <= comp_image[v, h]
            xor_condition = np.logical_xor(ref_condition, comp_condition)
            loe_loss += np.sum(xor_condition)
    
    return loe_loss / (total_pixels * 1000)


def generate_sparse_difference_matrices(illum_map: np.ndarray) -> (csr_matrix, csr_matrix):
    """
    Generates sparse Toeplitz matrices for computing the forward difference in both
    horizontal and vertical directions for the given illumination map.

    Args:
    - illum_map (np.ndarray): Input illumination map array of shape (M, N).

    Returns:
    - tuple: A tuple containing two csr_matrix objects for horizontal and vertical differences.
    """
    height, width = illum_map.shape
    total_elements = illum_map.size

    # Lists to hold row, column, and value data for sparse matrices
    dx_rows, dx_cols, dx_vals = [], [], []
    dy_rows, dy_cols, dy_vals = [], [], []

    for i in range(total_elements):
        if i + width < total_elements:
            dy_rows.extend([i, i])
            dy_cols.extend([i, i + width])
            dy_vals.extend([-1, 1])
        if (i + 1) % width != 0:
            dx_rows.extend([i, i])
            dx_cols.extend([i, i + 1])
            dx_vals.extend([-1, 1])

    # Create compressed sparse row matrices
    dx_matrix = csr_matrix((dx_vals, (dx_rows, dx_cols)), shape=(total_elements, total_elements))
    dy_matrix = csr_matrix((dy_vals, (dy_rows, dy_cols)), shape=(total_elements, total_elements))

    return dx_matrix, dy_matrix


def compute_partial_derivative(matrix: np.ndarray, sparse_matrix: csr_matrix) -> np.ndarray:
    """
    Calculates the partial derivative of a matrix using a given Toeplitz sparse matrix.

    Args:
    - matrix (np.ndarray): Input matrix of shape (M, N).
    - sparse_matrix (csr_matrix): Toeplitz sparse matrix for derivative computation.

    Returns:
    - np.ndarray: Array of derivative values with shape (M, N).
    """
    # Flatten the input matrix and compute the product with the sparse matrix
    flattened_matrix = matrix.flatten()
    derivative_flat = sparse_matrix.dot(flattened_matrix)

    # Reshape the result back to the original matrix shape
    derivative_matrix = derivative_flat.reshape(matrix.shape)

    return derivative_matrix


def compute_gaussian_weights(gradient: np.ndarray, kernel_size: int, sigma: Union[int, float], epsilon: float) -> np.ndarray:
    """
    Computes the weight matrix according to the third weight strategy of the original LIME paper.

    Args:
    - gradient (np.ndarray): Gradient matrix of shape (M, N).
    - kernel_size (int): Size of the Gaussian kernel.
    - sigma (Union[int, float]): Standard deviation for the Gaussian kernel.
    - epsilon (float): Small constant to prevent division by zero.

    Returns:
    - np.ndarray: Weight matrix of shape (M, N).
    """
    radius = (kernel_size - 1) // 2

    # Compute the absolute value of the gradient
    abs_gradient = np.abs(gradient)
    
    # Apply the Gaussian filter to the absolute gradient
    smoothed_gradient = gaussian_filter(abs_gradient, sigma, radius=radius, mode='constant')
    
    # Compute the denominator with epsilon to avoid division by zero
    denominator = epsilon + smoothed_gradient
    
    # Invert the denominator
    inverted_denominator = 1 / denominator
    
    # Apply the Gaussian filter to the inverted denominator
    weights = gaussian_filter(inverted_denominator, sigma, radius=radius, mode='constant')
    
    return weights


def initialize_weight_matrices(illum_map: np.ndarray, strategy: int, epsilon: float = 0.001) -> (np.ndarray, np.ndarray):
    """
    Initializes weight matrices according to a chosen strategy from the original LIME paper.
    Then updates and vectorizes these weight matrices to prepare them for calculation of a new illumination map.

    Args:
    - illum_map (np.ndarray): Illumination map of shape (M, N).
    - strategy (int): Strategy number (1, 2, or 3) for weight initialization.
    - epsilon (float): Small constant to avoid division by zero.

    Returns:
    - tuple: Flattened weight matrices for horizontal and vertical directions.
    """
    if strategy == 1:
        weights_x = np.ones(illum_map.shape)
        weights_y = np.ones(illum_map.shape)
    elif strategy == 2:
        d_x, d_y = generate_sparse_difference_matrices(illum_map)
        grad_x = compute_partial_derivative(illum_map, d_x)
        grad_y = compute_partial_derivative(illum_map, d_y)
        weights_x = 1 / (np.abs(grad_x) + epsilon)
        weights_y = 1 / (np.abs(grad_y) + epsilon)
    else:
        sigma = 2
        size = 15
        d_x, d_y = generate_sparse_difference_matrices(illum_map)
        grad_x = compute_partial_derivative(illum_map, d_x)
        grad_y = compute_partial_derivative(illum_map, d_y)
        weights_x = compute_gaussian_weights(grad_x, size, sigma, epsilon)
        weights_y = compute_gaussian_weights(grad_y, size, sigma, epsilon)

    # Normalize the weights
    normalized_weights_x = weights_x / (np.abs(grad_x) + epsilon)
    normalized_weights_y = weights_y / (np.abs(grad_y) + epsilon)

    # Flatten the weight matrices
    flattened_weights_x = normalized_weights_x.flatten()
    flattened_weights_y = normalized_weights_y.flatten()

    return flattened_weights_x, flattened_weights_y


def refine_illumination_map(illumination_map: np.ndarray, strategy: int = 3) -> np.ndarray:
    """
    Refines the initial illumination map using a sped-up solver as described in the original LIME paper.

    Args:
    - illumination_map (np.ndarray): Initial illumination map of shape (M, N).
    - strategy (int): Weight strategy to use (1, 2, or 3). Default is 3.

    Returns:
    - np.ndarray: Refined illumination map of shape (M, N).
    """
    # Reshape the illumination map into a vector
    vectorized_map = illumination_map.reshape((-1, 1))
    
    # Constants
    epsilon = 0.001
    alpha = 0.15

    # Generate sparse difference matrices
    diff_x_sparse, diff_y_sparse = generate_sparse_difference_matrices(illumination_map)

    # Initialize weight matrices
    weight_x_flat, weight_y_flat = initialize_weight_matrices(illumination_map, strategy, epsilon)

    # Create diagonal matrices from weight vectors
    diag_weight_x = diags(weight_x_flat)
    diag_weight_y = diags(weight_y_flat)

    # Compute the x and y terms for the matrix
    x_term = diff_x_sparse.T.dot(diag_weight_x).dot(diff_x_sparse)
    y_term = diff_y_sparse.T.dot(diag_weight_y).dot(diff_y_sparse)

    # Create identity matrix
    identity_matrix = diags(np.ones(x_term.shape[0]))

    # Construct the matrix for the linear system
    system_matrix = identity_matrix + alpha * (x_term + y_term)

    # Solve the linear system to update the illumination map
    updated_vector = spsolve(csr_matrix(system_matrix), vectorized_map)

    # Reshape the updated vector back to the original illumination map shape
    refined_illumination_map = updated_vector.reshape(illumination_map.shape)

    return refined_illumination_map



def perform_bm3d_denoising(input_image: np.ndarray, illum_correction_map: np.ndarray, noise_std: Union[int, float] = 0.02) -> np.ndarray:
    """Applies BM3D denoising to the Y channel of an input image and adjusts its brightness using an illumination correction map.

    Returns a denoised and brightness-corrected image with pixel intensities clipped to a maximum of 1.
    """
    # Convert the input image from RGB to YUV color space
    yuv_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2YUV)
    
    # Extract the Y (luminance) channel from the YUV image
    luminance_channel = yuv_image[:, :, 0]
    
    # Perform BM3D denoising on the Y channel
    denoised_luminance = bm3d(luminance_channel, noise_std)
    
    # Replace the Y channel in the YUV image with the denoised Y channel
    yuv_image[:, :, 0] = denoised_luminance
    
    # Convert the YUV image back to RGB color space
    denoised_rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
    
    # Combine the input image and the denoised image using the illumination correction map
    final_image = input_image * illum_correction_map + denoised_rgb_image * (1 - illum_correction_map)
    
    # Clip the pixel values to the range [0, 1] and convert to float32
    return np.clip(final_image, 0, 1).astype("float32")
