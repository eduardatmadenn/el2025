import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the two images
img1 = cv2.imread('1.png', cv2.IMREAD_COLOR)  # Image 1
img2 = cv2.imread('2.png', cv2.IMREAD_COLOR)  # Image 2

# Load the logo with a transparent background
logo = cv2.imread('task_1/logo.png', cv2.IMREAD_UNCHANGED)  # Logo with transparency

# Get the dimensions of the images and the logo
h_img1, w_img1 = img1.shape[:2]
h_img2, w_img2 = img2.shape[:2]
h_logo, w_logo = logo.shape[:2]

# Calculate the position to place the logo (centered)
center_x = (w_img1 - w_logo) // 2
center_y = (h_img1 - h_logo) // 2

# Ensure the logo has 4 channels (including alpha)
if logo.shape[2] == 4:
    # Separate the RGBA channels
    alpha_logo = logo[:, :, 3] / 255.0  # Alpha channel (transparency)
    logo_rgb = logo[:, :, :3]  # RGB channels of the logo

    # Place the logo on Image 1
    for c in range(0, 3):  # For each channel (R, G, B)
        img1[center_y:center_y+h_logo, center_x:center_x+w_logo, c] = (
            alpha_logo * logo_rgb[:, :, c] + 
            (1 - alpha_logo) * img1[center_y:center_y+h_logo, center_x:center_x+w_logo, c]
        )

    # Place the logo on Image 2
    for c in range(0, 3):
        img2[center_y:center_y+h_logo, center_x:center_x+w_logo, c] = (
            alpha_logo * logo_rgb[:, :, c] + 
            (1 - alpha_logo) * img2[center_y:center_y+h_logo, center_x:center_x+w_logo, c]
        )

# Save the images with the logo added
cv2.imwrite("task_1_1.png", img1)
cv2.imwrite("task_1_2.png", img2)
