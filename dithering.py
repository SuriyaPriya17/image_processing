import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def ordered_dithering(image, matrix):
    h, w = image.shape
    dithered = np.zeros((h, w), dtype=np.uint8)
    matrix_size = matrix.shape[0]

    for i in range(h):
        for j in range(w):
            threshold = matrix[i % matrix_size, j % matrix_size] * 255 / (matrix_size**2)
            dithered[i, j] = 255 if image[i, j] > threshold else 0

    return dithered

def error_diffusion_halftoning(image):
    h, w = image.shape
    halftoned = np.zeros((h, w), dtype=np.uint8)
    error = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            old_pixel = image[i, j] + error[i, j]
            new_pixel = 255 if old_pixel > 128 else 0
            halftoned[i, j] = new_pixel
            quant_error = old_pixel - new_pixel

            if j + 1 < w:
                error[i, j + 1] += quant_error * 7 / 16
            if i + 1 < h:
                if j > 0:
                    error[i + 1, j - 1] += quant_error * 3 / 16
                error[i + 1, j] += quant_error * 5 / 16
                if j + 1 < w:
                    error[i + 1, j + 1] += quant_error * 1 / 16

    return halftoned

image_path = 'flowers.jpg'
image = Image.open(image_path).convert('L')
image_np = np.array(image)

bayer_matrix = np.array([[0, 2],
                          [3, 1]])

ordered_dithered_image = ordered_dithering(image_np, bayer_matrix)
error_diffused_image = error_diffusion_halftoning(image_np)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Grayscale Image')
plt.imshow(image_np, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Ordered Dithering')
plt.imshow(ordered_dithered_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Error Diffusion Halftoning')
plt.imshow(error_diffused_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

