import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

h, w = 500, 500

# Step 1: Create the base white image
img = np.ones((h, w), dtype=np.uint8) * 255

# Step 2: Add text in the bottom-right corner
text_color = 248
font = cv2.FONT_HERSHEY_SIMPLEX
text = "CTF{3m2t7b}"
font_scale = 0.3
thickness = 1
text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
text_x = w - text_size[0] - 10
text_y = h - 10
cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

# Step 3: Create noise maps based on Gaussian distributions
y, x = np.indices((h, w))
mask_prob = (x + y) / (2 * max(h, w))
random_matrix = np.random.rand(h, w)
mask = random_matrix < mask_prob
noise1 = np.random.normal(loc=240, scale=2, size=(h, w))
noise2 = np.random.normal(loc=250, scale=2, size=(h, w))

# Step 4: Add the generated noise to the base image
noise_combined = np.where(mask, noise2, noise1)
img_noise = img.astype(np.float32) + (noise_combined - 255)
img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)

# Step 5: Save the image with text and noise
cv2.imwrite('1.png', img_noise)

# Step 6: Create an image without text (only noise)
img_no_text = np.ones((h, w), dtype=np.uint8) * 255
noise_combined_no_text = np.where(mask, noise2, noise1)
img_noise_no_text = img_no_text.astype(np.float32) + (noise_combined_no_text - 255)
img_noise_no_text = np.clip(img_noise_no_text, 0, 255).astype(np.uint8)

# Step 7: Save the image without text and noise
cv2.imwrite('2.png', img_noise_no_text)

# Step 8: Detect differences between the two images
difference = cv2.absdiff(img_noise, img_noise_no_text)

# Step 9: Convert difference to grayscale if not already
if len(difference.shape) == 3:
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
else:
    difference_gray = difference

# Step 10: Binarize the difference using Otsu's method
_, binarized_diff = cv2.threshold(difference_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 11: Display the binarized difference (text detection highlight)
plt.figure(figsize=(6,6))
plt.imshow(binarized_diff, cmap='gray')
plt.title("Binarized Difference (Text Detection)")
plt.axis('off')
plt.show()
