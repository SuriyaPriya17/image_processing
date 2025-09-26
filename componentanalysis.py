import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('solderjoints.jpg', cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

min_good_area = 50
max_good_area = 300

good_count = 0
defective_count = 0

output_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for label in range(1, num_labels):
    x, y, w, h, area = stats[label]

    if min_good_area <= area <= max_good_area:
        good_count += 1
        color = (0, 255, 0)
    else:
        defective_count += 1
        color = (0, 0, 255)

    cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(output_img, f'{area}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
print(f"Total solder joints detected: {num_labels - 1}")
print(f"Good solder joints: {good_count}")
print(f"Defective solder joints: {defective_count}")

plt.figure(figsize=(12, 10))
plt.title('Solder Joint Classification')
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
