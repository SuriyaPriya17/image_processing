import cv2
import numpy as np
import matplotlib.pyplot as plt
product_img = cv2.imread("product.jpg", cv2.IMREAD_GRAYSCALE)
template_img = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(5000)

kp1, des1 = orb.detectAndCompute(template_img, None)
kp2, des2 = orb.detectAndCompute(product_img, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

orb_match_img = cv2.drawMatches(template_img, kp1, product_img, kp2, matches[:30], None, flags=2)

res = cv2.matchTemplate(product_img, template_img, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

h, w = template_img.shape
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
template_match_img = cv2.cvtColor(product_img, cv2.COLOR_GRAY2BGR)
cv2.rectangle(template_match_img, top_left, bottom_right, (0,0,255), 2)

plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.imshow(orb_match_img)
plt.title("ORB Feature Matching")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(template_match_img, cmap='gray')
plt.title(f"Template Matching (score={max_val:.2f})")
plt.axis("off")

plt.show()