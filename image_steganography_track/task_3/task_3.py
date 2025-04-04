import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
import string


# Function to generate a random flag for CTF
def generate_flag():
    charset = string.ascii_letters + string.digits  # Include both lowercase, uppercase letters and digits
    parts = ["".join(random.choices(charset, k=3)) for _ in range(3)]
    return f"CTF{{{'_'.join(parts)}}}"


# Function to add noise to the image
def add_noise(image, noise_type="gaussian"):
    if noise_type == "gaussian":
        mean = 0
        stddev = 20  # Adjust the noise level
        noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
        noisy_img = cv2.add(image, noise)
    
    elif noise_type == "salt_pepper":
        s_vs_p = 0.5  # Proportion of salt to pepper noise
        amount = 0.02  # Noise intensity
        noisy_img = image.copy()

        num_salt = np.ceil(amount * image.size * s_vs_p)
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))

        # Add salt (white pixels)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_img[tuple(coords)] = 255

        # Add pepper (black pixels)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_img[tuple(coords)] = 0

    elif noise_type == "speckle":
        noise = np.random.randn(*image.shape) * 0.2 * 255
        noisy_img = image + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    else:
        noisy_img = image  # No noise
    
    return noisy_img


if __name__ == "__main__":
    input_img_path = 'original_steganography.png'
    output_path = 'task_3.png'
    solve = True
    gen_flag = False

    # Generate flag or use the predefined one
    if gen_flag:
        flag_text = generate_flag()
        print(flag_text)
    else:
        flag_text = "CTF{aT2_Zv7_RdB}"
    
    # Load the image for steganography
    img = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED)
    h, w, _  = img.shape

    # Define the region where the text will be hidden
    y_offset, x_offset = 100, 100
    box_h, box_w = 50, 200  # Text box dimensions

    # Ensure the region is within the image boundaries
    y1, y2 = max(0, y_offset), min(h, y_offset + box_h)
    x1, x2 = max(0, x_offset), min(w, x_offset + box_w)

    # Calculate the average color in the region for text insertion (blue channel only)
    region = img[y1:y2, x1:x2, 2]
    mean_color = np.mean(region, axis=(0, 1))

    # Add Gaussian noise to the mean color
    noise_std = 20
    noisy_color = mean_color + np.random.normal(0, noise_std, 3)
    noisy_color = np.clip(noisy_color, 0, 255).astype(np.uint8)

    # Create a text layer with transparent background
    text_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)

    # Choose font
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Generate the flag text
    text_width = draw.textlength(flag_text, font=font)
    text_position = ((x1 + x2 - text_width) // 2, (y1 + y2 - 20) // 2)

    # Draw the text with noisy color and transparency
    draw.text(text_position, flag_text, fill=(int(noisy_color[0]), int(noisy_color[1]), int(noisy_color[2]), 50), font=font)

    # Convert the text layer to a NumPy array
    text_layer_np = np.array(text_layer)

    # Integrate the text into the original image
    for y in range(h):
        for x in range(w):
            alpha = text_layer_np[y, x, 3] / 255.0  # Transparency of the pixel
            img[y, x, 0] = int(img[y, x, 0] * (1 - alpha) + text_layer_np[y, x, 2] * alpha)  # Modify only the blue channel

    # Apply Gaussian Blur to the image before saving
    img = cv2.GaussianBlur(img, (7, 7), 1.3)

    # Save the modified image
    cv2.imwrite(output_path, img)

    if solve:
        # Decoding: Apply sharpening filter to highlight the text
        blue_channel = img[:, :, 0]
        # sharpen_kernel = np.array([[0, -1, 0], 
        #                            [-1, 5, -1], 
        #                            [0, -1, 0]])  # Sharpening kernel
        # sharpen_kernel = np.ones((3, 3)) * (-8)
        # sharpen_kernel[1, 1] = 9
        sharpen_kernel = np.array([[0,  0, -1,  0,  0],
                            [0, -1, -2, -1,  0],
                            [-1, -2, 17, -2, -1],
                            [0, -1, -2, -1,  0],
                            [0,  0, -1,  0,  0]])  # Sharpening kernel for edge enhancement
        blue_channel_sharp = cv2.filter2D(blue_channel, -1, sharpen_kernel)
        cv2.imwrite('task_3_no_sharpen.png', blue_channel)
        cv2.imwrite('task_3_solved.png', blue_channel_sharp)