import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('metal_surface.jpg', cv2.IMREAD_GRAYSCALE)
_, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
adaptive_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
adaptive_gauss = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.title('Original Metal Surface')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Global Thresholding')
plt.imshow(global_thresh, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Adaptive Mean Thresholding')
plt.imshow(adaptive_mean, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Adaptive Gaussian Thresholding')
plt.imshow(adaptive_gauss, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Otsu's Thresholding")
plt.imshow(otsu_thresh, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
