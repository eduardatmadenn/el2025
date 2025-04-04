import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def harta_de_transformare(r, L):
    #initializare harta (tineti cont ca harta din imshow data ca parametru este vectoriala nu scalara)
    #completati voi dimensiunea hartii
    h=np.zeros(L)

    #Parcurg fiecare nivel de cuantizare(instensitate de gri) posibil calculez noua valoare folosind ecuatia 2.1
    #completati functia range cu numarul maxim de intensitati de gri
    # cand completati harta tineti cont ca este o matrice cu 3 coloane si numarul de linii egal cu numarul de intensitati de gri
    # NU Uitati ca pentru o harta "gray" valorile pentru un triplet(o linie din h) sunt egale (R=G=B)
    for i in range(L):
        h[i] = (L - 1) * ((i / (L - 1)) ** r)
    return h

def harta_de_transformare_inversa(h, L, r):
    # Inversarea transformarii
    h_invers = np.zeros_like(h)

    for i in range(L):
        # Calculăm inversa fiecărei valori din h
        h_invers[i] = (L - 1) * ((h[i] / (L - 1)) ** (1 / r))

    # Se asigură că valorile sunt în intervalul [0, 255] (pentru a fi corespunzătoare unei imagini 8-bit)
    # h_invers = np.clip(h_invers, 0, 255)
    return h_invers


def add_hidden_text(image, text, font_size=20, noise_std=20, y_offset=100, x_offset=100):
    # Asigură-te că imaginea este în tonuri de gri
    if len(image.shape) == 3:  # Dacă imaginea este color (RGB), o convertim în grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Obținem dimensiunile imaginii
    h, w = image.shape
    
    # Creăm un layer transparent pentru text
    text_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    
    # Încearcă să folosești un font standard, altfel folosește fontul default
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculăm lățimea textului și poziția acestuia
    text_width = draw.textlength(text, font=font)
    text_position = ((w - text_width) // 2, y_offset)
    
    # Convertim text_width și font_size la întregi pentru a le putea folosi în slicing
    text_width = int(text_width)
    font_size = int(font_size)
    
    # Calculăm culoarea medie din zona unde vom adăuga textul
    region = image[y_offset:y_offset + font_size, x_offset:x_offset + text_width]
    mean_color = np.mean(region)

    # Adăugăm zgomot gaussian pe culoarea medie
    noisy_color = mean_color + np.random.normal(0, noise_std)
    noisy_color = np.clip(noisy_color, 0, 255).astype(np.uint8)
    # print(noisy_color)
    noisy_color = 105
    
    # Desenăm textul cu culoarea zgomotoasă
    draw.text(text_position, text, fill=(noisy_color, noisy_color, noisy_color, 120), font=font)
    
    # Convertim layer-ul de text la un array numpy
    text_layer_np = np.array(text_layer)
    
    # Ascundem textul în imagine (utilizăm transparența pentru a face blending)
    for y in range(h):
        for x in range(w):
            alpha = text_layer_np[y, x, 3] / 255.0
            image[y, x] = int(image[y, x] * (1 - alpha) + text_layer_np[y, x, 0] * alpha)  # Folosim același canal pentru gri
    
    return image

def apply_map(image, contrast_map):   
    h, w = image.shape
    # Apply the transformation to the image (overexposure effect)
    for i in range(h):
        for j in range(w):
            image[i, j] = int(contrast_map[image[i, j]])
    return image


# Load original image
L, r = 256, 0.15
img = cv2.imread('image_steganography/test_steganography.png', cv2.IMREAD_GRAYSCALE)

# Add hidden text and apply overexpose effect
text = "CTF{hidd3n_fl4g}"
img_with_text = add_hidden_text(img.copy(), text)
cv2.imwrite('image_steganography/contrast/text_image.png', img_with_text)

# Apply overexpose effect to the image
h_direct = harta_de_transformare(r=r, L=L)
img_overexposed = apply_map(img_with_text.copy(), h_direct)

# Save the overexposed image with hidden text
cv2.imwrite('image_steganography/contrast/overexposed_image.png', img_overexposed)

# Decoding: Apply inverse transformation to extract text
img_decoded = img_overexposed.copy()
h_direct_2 = harta_de_transformare(r=3, L=L)
# h_invers = harta_de_transformare_inversa(h_direct, L, r)
img_decoded = apply_map(img_decoded, h_direct_2)

# Show the decoded image
cv2.imwrite('image_steganography/contrast/decoded_image.png', img_decoded)