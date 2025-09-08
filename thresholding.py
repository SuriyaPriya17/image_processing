import cv2
import matplotlib.pyplot as plt
image_path = "D:\\wood\\test\\color\\003.png" 
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
if img is None:
    print("Error: Could not load image.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, trunc = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
_, tozero = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
titles = ["Original (Gray)", "BINARY", "TRUNC", "TOZERO"]
images = [gray, binary,trunc, tozero]

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 3, i+1)
    cmap = 'gray' if len(images[i].shape) == 2 else None
    plt.imshow(images[i], cmap=cmap)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
