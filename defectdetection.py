import cv2
import matplotlib.pyplot as plt

img = cv2.imread("D:\\wood\\test\\scratch\\005.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (31,31), 0)
illum_corrected = cv2.divide(gray, blur, scale=255)
_, otsu_mask = cv2.threshold(illum_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptive_mask = cv2.adaptiveThreshold(illum_corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 35, 5)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1);
plt.imshow(otsu_mask, cmap='gray');
plt.title('Otsu Threshold')
plt.subplot(1,2,2);
plt.imshow(adaptive_mask, cmap='gray');
plt.title('Adaptive Threshold')
plt.show()
