import cv2
import matplotlib.pyplot as plt

image = cv2.imread("mountain.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

H, W, _ = image.shape

focal_lengths = [24, 35, 50, 85, 135]
base_focal = 24  

def simulate_fov(img, base_focal, target_focal):
     
    scale = base_focal / target_focal 
    h, w, _ = img.shape
    new_w, new_h = int(w * scale), int(h * scale)

    x1 = max(0, (w - new_w) // 2)
    y1 = max(0, (h - new_h) // 2)

    cropped = img[y1:y1+new_h, x1:x1+new_w]

    simulated = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)

    return simulated

plt.figure(figsize=(15,5))

for i, f in enumerate(focal_lengths):
    sim = simulate_fov(image, base_focal, f)
    plt.subplot(1, len(focal_lengths), i+1)
    plt.imshow(sim)
    plt.title(f"{f}mm")
    plt.axis('off')

plt.suptitle("Results")
plt.show()
