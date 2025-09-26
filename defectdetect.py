import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('glass_bottle.jpg', cv2.IMREAD_GRAYSCALE)

denoised = cv2.GaussianBlur(image, (5, 5), 0)
edges = cv2.Canny(denoised, threshold1=50, threshold2=150)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 100
defect_bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]
output_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in defect_bounding_boxes:
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Edge Detection + Morphology')
plt.imshow(opened, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Defect Localization')
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
