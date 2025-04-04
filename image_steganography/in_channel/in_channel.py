import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


# ## Hidden channel
# # Load the original image
# img = cv2.imread('image_steganography/test_steganography.png', cv2.COLOR_BGR2RGB)
# h, w, _  = img.shape
# blue_img = img[:, :, 2]
# blue_img = Image.fromarray(blue_img)

# # Create the flag image (black text on white background)
# # flag_img = cv2.imread('image_steganography/flag.png', cv2.IMREAD_GRAYSCALE)
# # Resize flag if needed to match a portion of the main image
# # flag_img = cv2.resize(flag_img, (200, 50))

# # Initialize drawing context
# draw = ImageDraw.Draw(blue_img)

# # Choose a font and size
# try:
#     # Try to use a common font
#     font = ImageFont.truetype("arial.ttf", 20)
# except IOError:
#     # Fallback to default font if arial isn't available
#     font = ImageFont.load_default()

# # Draw the flag text in black
# flag_text = "CTF{hidd3n_fl4g}"
# text_width = draw.textlength(flag_text, font=font)
# text_position = ((w - text_width) // 4, (h - text_width) // 4)  # Center the text horizontally
# draw.text(text_position, flag_text, fill=0, font=font)  # 0 = black

# # Choose region to hide the flag
# y_offset, x_offset = 100, 100
# blue_img =  np.array(blue_img)

# # Hide flag in only the blue channel
# for y in range(h):
#     for x in range(w):
#         if 0 <= y_offset + y < img.shape[0] and 0 <= x_offset + x < img.shape[1]:
#             # Only modify blue channel (index 0 in OpenCV's BGR format)
#             # Scale down the flag's intensity to make it less visible
#             img[y_offset + y, x_offset + x, 0] = (
#                 img[y_offset + y, x_offset + x, 0] * 0.8 + 
#                 blue_img[y, x] * 0.2
#             )

# cv2.imwrite('image_steganography/channel_hidden.png', img)

# # To extract:
# blue_channel = img[:,:,0]
# cv2.imwrite('image_steganography/blue_channel.png', blue_channel)

# Funcție pentru adăugarea zgomotului
def add_noise(image, noise_type="gaussian"):
    if noise_type == "gaussian":
        mean = 0
        stddev = 20  # Ajustează nivelul de zgomot
        noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
        noisy_img = cv2.add(image, noise)
    
    elif noise_type == "salt_pepper":
        s_vs_p = 0.5  # Proporția dintre sare și piper
        amount = 0.02  # Intensitatea zgomotului
        noisy_img = image.copy()

        num_salt = np.ceil(amount * image.size * s_vs_p)
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))

        # Adaugă sare (pixeli albi)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_img[tuple(coords)] = 255

        # Adaugă piper (pixeli negri)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_img[tuple(coords)] = 0

    elif noise_type == "speckle":
        noise = np.random.randn(*image.shape) * 0.2 * 255
        noisy_img = image + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    else:
        noisy_img = image  # Fără zgomot
    
    return noisy_img


img = cv2.imread('image_steganography/test_steganography.png', cv2.IMREAD_UNCHANGED)
h, w, _  = img.shape

# Zona unde vom ascunde textul
y_offset, x_offset = 100, 100
box_h, box_w = 50, 200  # Dimensiunea zonei de text

# Asigură-te că zona este în limitele imaginii
y1, y2 = max(0, y_offset), min(h, y_offset + box_h)
x1, x2 = max(0, x_offset), min(w, x_offset + box_w)

# Calculează culoarea medie în zona de inserare a textului (doar canalul albastru)
region = img[y1:y2, x1:x2, 2]
mean_color = np.mean(region, axis=(0, 1))

# Adaugă zgomot gaussian pe culoarea medie
noise_std = 20
noisy_color = mean_color + np.random.normal(0, noise_std, 3)
noisy_color = np.clip(noisy_color, 0, 255).astype(np.uint8)

# Creăm imaginea de text cu fundal transparent
text_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
draw = ImageDraw.Draw(text_layer)

# Alege fontul
try:
    font = ImageFont.truetype("arial.ttf", 15)
except IOError:
    font = ImageFont.load_default()

# Text și poziționare
flag_text = "CTF{x2_yz4_flo}"
text_width = draw.textlength(flag_text, font=font)
text_position = ((x1 + x2 - text_width) // 2, (y1 + y2 - 20) // 2)

# Desenează textul cu culoarea zgomotoasă și transparență
draw.text(text_position, flag_text, fill=(int(noisy_color[0]), int(noisy_color[1]), int(noisy_color[2]), 50), font=font)

# Convertim text_layer la NumPy
text_layer_np = np.array(text_layer)

# Integrează textul în imaginea originală
for y in range(h):
    for x in range(w):
        alpha = text_layer_np[y, x, 3] / 255.0  # Transparanța pixelului
        img[y, x, 0] = int(img[y, x, 0] * (1 - alpha) + text_layer_np[y, x, 2] * alpha)  # Modifică doar canalul albastru

# **Aplică un Gaussian Blur pe imagine înainte de salvare**
img = cv2.GaussianBlur(img, (5, 5), 1.3)

# Salvăm imaginea modificată
cv2.imwrite('image_steganography/channel_hidden.png', img)

# **Decodare: Aplică un filtru de ascuțire pentru a evidenția textul**
blue_channel = img[:, :, 0]
sharpen_kernel = np.array([[0, -1, 0], 
                           [-1, 5, -1], 
                           [0, -1, 0]])  # Kernel pentru ascuțire
# sharpen_kernel = np.array([[0,  0, -1,  0,  0],
#                        [0, -1, -2, -1,  0],
#                        [-1, -2, 16, -2, -1],
#                        [0, -1, -2, -1,  0],
#                        [0,  0, -1,  0,  0]])
blue_channel_sharp = cv2.filter2D(blue_channel, -1, sharpen_kernel)
img_sharp = cv2.filter2D(img, -1, sharpen_kernel)
cv2.imwrite('image_steganography/blue_channel.png', blue_channel)
cv2.imwrite('image_steganography/blue_channel_sharpened.png', blue_channel_sharp)
cv2.imwrite('image_steganography/noisy_img_sharpened.png', img_sharp)

# # Filtru Sobel pentru evidențierea contururilor
# sobel_x = cv2.Sobel(blue_channel, cv2.CV_64F, 2, 0, ksize=5)
# sobel_y = cv2.Sobel(blue_channel, cv2.CV_64F, 0, 2, ksize=5)
# sobel_combined = cv2.magnitude(sobel_x, sobel_y)
# cv2.imwrite('image_steganography/blue_channel_sobel.png', sobel_combined)