import cv2
import numpy as np
import matplotlib.pyplot as plt
def morphology_ops(img_path):
    img = cv2.imread(img_path, 0)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1); plt.imshow(binary, cmap='gray');
    plt.title("Binary")
    plt.subplot(1,3,2); plt.imshow(opening, cmap='gray');
    plt.title("Opening")
    plt.subplot(1,3,3); plt.imshow(closing, cmap='gray');
    plt.title("Closing")
    plt.show()

if __name__ == "__main__":
    morphology_ops("C:\\Users\\priya\\OneDrive\\Pictures\\Screenshots\\Screenshot 2025-09-08 201620.png")
