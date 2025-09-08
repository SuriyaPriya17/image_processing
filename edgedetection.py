import cv2
import matplotlib.pyplot as plt

img = cv2.imread("D:\\screw\\test\\scratch_neck\\012.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  
sobel_mag = cv2.magnitude(sobelx, sobely)
sobel_mag = cv2.convertScaleAbs(sobel_mag)
edges = cv2.Canny(gray, 100, 150)
plt.figure(figsize=(12,6))
plt.subplot(1,3,1);
plt.imshow(gray, cmap='gray');
plt.title('Grayscale')
plt.subplot(1,3,2);
plt.imshow(sobel_mag, cmap='gray');
plt.title('Sobel')
plt.subplot(1,3,3);
plt.imshow(edges, cmap='gray');
plt.title('Canny')
plt.show()
