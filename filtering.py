import cv2
import matplotlib.pyplot as plt
import numpy as np
def filters(img_path):
    img = cv2.imread(img_path)
    blur = cv2.GaussianBlur(img,(5,5), 0)
    median = cv2.medianBlur(img,5)
    sharpen = cv2.filter2D(img, -1,kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], dtype=np.float32))
    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Original")
    plt.subplot(2,2,2); plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)); plt.title("Gaussian Blur")
    plt.subplot(2,2,3); plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB)); plt.title("Median Blur")
    plt.subplot(2,2,4); plt.imshow(cv2.cvtColor(sharpen, cv2.COLOR_BGR2RGB)); plt.title("Sharpen")
    plt.show()

if __name__ == "__main__":
    filters("D:\\transistor\\test\\good\\023.png")
