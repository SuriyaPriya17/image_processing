import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, img, cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')

def main(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Error: Image not found.")
        return
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    _,binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1); show_image("(RGB)",rgb)
    plt.subplot(2, 3, 2); show_image("Grayscale", gray, cmap='gray')
    plt.subplot(2, 3, 3); show_image("HSV", cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    plt.subplot(2, 3, 4); show_image("LAB", cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    plt.subplot(2, 3, 5); show_image("Binary", binary, cmap='gray')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main("C:\\Users\\priya\\OneDrive\\Pictures\\Screenshots\\Screenshot 2025-09-08 201620.png")
