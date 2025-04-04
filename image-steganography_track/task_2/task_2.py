import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Unlinear contrast adjustment function
def transform_map(r, L):
    h=np.zeros(L)
    for i in range(L):
        h[i] = (L - 1) * ((i / (L - 1)) ** r)  # Apply non-linear mapping based on the power parameter 'r'
    return h

# Function to add hidden text to an image (text steganography)
def add_hidden_text(image, text, font_size=20, noise_std=20, y_offset=100, x_offset=100):
# def add_hidden_text(image, text, font_size=15, noise_std=20, y_offset=30, x_offset=100):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    
    # Create an empty transparent layer for the text
    text_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    
    # Try to load a font, fallback to default if not found
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate the position of the text on the image (centered horizontally)
    text_width = draw.textlength(text, font=font)
    x_offset = int((w - text_width) // 2)
    text_position = (x_offset, y_offset)
    
    # Calculate mean color of the region where text will be placed
    region = image[y_offset:y_offset + font_size, x_offset:x_offset + int(text_width)]
    # mean_color = np.mean(region)
    noisy_color = 104

    # Draw the text with the noisy color on the transparent layer
    draw.text(text_position, text, fill=(noisy_color, noisy_color, noisy_color, 120), font=font)
    
    # Convert the text layer to a numpy array and blend with the original image
    text_layer_np = np.array(text_layer)
    for y in range(h):
        for x in range(w):
            alpha = text_layer_np[y, x, 3] / 255.0
            image[y, x] = int(image[y, x] * (1 - alpha) + text_layer_np[y, x, 0] * alpha)  # Apply blending using transparency
    
    return image

# Function to apply contrast adjustment (overexpose effect)
def apply_map(image, contrast_map):
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            image[i, j] = int(contrast_map[image[i, j]])  # Apply contrast mapping to each pixel
    return image

if __name__ == '__main__':
    orig_path = 'original_steganography.png'
    output_path = 'task_2.png'
    flag = 'CTF{2pL_grt}'  # Hidden text (flag)
    solve = True  # Flag to decide if decoding will be done

    # Load the original image as grayscale
    L, r = 256, 0.15
    img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)

    # Add hidden text to the image
    img_with_text = add_hidden_text(img.copy(), flag)

    # Apply overexpose effect to the image using the transformation map
    h_direct = transform_map(r=r, L=L)
    img_overexposed = apply_map(img_with_text.copy(), h_direct)
    img_overexposed.clip(0, 255)

    # Save the overexposed image with hidden text
    cv2.imwrite(output_path, img_overexposed)

    if solve:
        # Decode the hidden text by applying inverse transformation
        img_decoded = img_overexposed.copy()
        h_direct_2 = transform_map(r=4, L=L)  # Inverse map with a different parameter
        img_decoded = apply_map(img_decoded, h_direct_2)

        # Save and display the decoded image
        cv2.imwrite('task_2_solved.png', img_decoded)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # clahe_img = clahe.apply(img_overexposed)
        # cv2.imwrite("image_steganography_challenge/task_1_solved_2.png", clahe_img)

        # equalized = cv2.equalizeHist(img_overexposed)
        # cv2.imwrite("image_steganography_challenge/task_1_solved_3.png", equalized)

