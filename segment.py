import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('flowers.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(image)
plt.title('RGB Image')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(hsv_img)
plt.title('HSV Image')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(lab_img)
plt.title('LAB Image')
plt.axis('off')

plt.show()

lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

mask_hsv = cv2.inRange(hsv_img, lower_green, upper_green)
segmented_hsv = cv2.bitwise_and(image, image, mask=mask_hsv)


lab_a = lab_img[:,:,1]
lab_b = lab_img[:,:,2]

mask_lab = cv2.inRange(lab_a, 100, 150)
segmented_lab = cv2.bitwise_and(image, image, mask=mask_lab)

plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plt.imshow(mask_hsv, cmap='gray')
plt.title('HSV Mask')

plt.subplot(2,2,2)
plt.imshow(segmented_hsv)
plt.title('HSV Segmentation Result')

plt.subplot(2,2,3)
plt.imshow(mask_lab, cmap='gray')
plt.title('LAB Mask')

plt.subplot(2,2,4)
plt.imshow(segmented_lab)
plt.title('LAB Segmentation Result')

plt.tight_layout()
plt.show()
