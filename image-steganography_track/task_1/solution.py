import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Read and subtract
img1 = cv2.imread('task_1_1.png')
img2 = cv2.imread('task_1_2.png')

# Diff only
#  img3 = img1 - img2
# img4 = img2 - img1

# plt.figure()
# plt.title("Diff only: 1 - 2")
# plt.axis('off')
# plt.imshow(img3)
# plt.show

# plt.figure()
# plt.title("Diff only: 2 - 1")
# plt.axis('off')
# plt.imshow(img4)
# plt.show

# Step 8: Detect difference between the two images
difference = cv2.absdiff(img1, img2)
difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
uq, cnt = np.unique(difference_gray, return_counts=True)
print(uq)
print(cnt)

# # Plot the hist
# plt.figure(figsize=(6,4))
# plt.hist(difference_gray.ravel(), bins=256, range=(0, 255), alpha=0.7)
# plt.title("Histogram of Grayscale Image")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Frequency")
# plt.show()

# Step 10: Apply histogram-based binarization (Otsu's method)
_, binarized_diff = cv2.threshold(difference_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 11: Show the binary difference (this is where the text is highlighted)
plt.figure(figsize=(6,6))
plt.imshow(binarized_diff, cmap='gray')
plt.title("Binarized Difference (Text Detection)")
plt.axis('off')
plt.show()

