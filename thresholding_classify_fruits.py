import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
ripe_lower = np.array([20, 100, 100])
ripe_upper = np.array([35, 255, 255])

unripe_lower = np.array([35, 50, 50])
unripe_upper = np.array([85, 255, 255])

def classify_fruit(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ripe_mask = cv2.inRange(hsv, ripe_lower, ripe_upper)
    unripe_mask = cv2.inRange(hsv, unripe_lower, unripe_upper)
    ripe_percent = np.sum(ripe_mask > 0) / (ripe_mask.shape[0] * ripe_mask.shape[1])
    unripe_percent = np.sum(unripe_mask > 0) / (unripe_mask.shape[0] * unripe_mask.shape[1])

    classification = "Unknown"
    if ripe_percent > 0.1:
        classification = "Ripe"
    elif unripe_percent > 0.1:
        classification = "Unripe"
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Ripe Mask')
    plt.imshow(ripe_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Unripe Mask')
    plt.imshow(unripe_mask, cmap='gray')
    plt.axis('off')

    plt.suptitle(f'Classification: {classification}')
    plt.show()
sample_images = [
    'banana_unripejpg.jpg',
    'ripe_banana.jpg',
    'ripe_orange.jpg',
    'unripe_mango.jpg',
    'unripe_orange.jpg'
]

for img_path in sample_images:
    classify_fruit(img_path)
