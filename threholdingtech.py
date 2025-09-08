import cv2
import matplotlib.pyplot as plt
def thresholding_methods(img_path):
    img = cv2.imread(img_path, 0)
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1); plt.imshow(img, cmap='gray');
    plt.title("Original")
    plt.subplot(1,3,2); plt.imshow(otsu, cmap='gray');
    plt.title("Otsu's Threshold")
    plt.subplot(1,3,3); plt.imshow(adaptive, cmap='gray');
    plt.title("Adaptive Threshold")
    plt.show()

if __name__ == "__main__":
    thresholding_methods("D:\\wood\\test\\color\\003.png")
