import cv2
import numpy as np
import os
import subprocess
from pathlib import Path
import shutil
from skimage.metrics import structural_similarity as ssim

import cv2
import numpy as np
import random

from bubble_SC_solve import *

def generate_4k_ball_grid(output_path='ball_grid_4k.png', ball_radius=10, padding=4, seed=41):
    
    random.seed(seed)
    np.random.seed(seed)
    
    
    target_res = (1080, 1920)  # height, width
    ball_diameter = ball_radius * 2

    rows = (target_res[0] - padding) // (ball_diameter + padding)
    cols = (target_res[1] - padding) // (ball_diameter + padding)

    h = rows * (ball_diameter + padding) + padding
    w = cols * (ball_diameter + padding) + padding
    image = np.full((h, w, 3), 40, dtype=np.uint8)

    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (0, 255, 255), (255, 255, 0), (255, 0, 255),
        (0, 128, 255), (147, 20, 255), (0, 200, 200), (200, 200, 0)
    ]

    for row in range(rows):
        for col in range(cols):
            x = padding + col * (ball_diameter + padding) + ball_radius
            y = padding + row * (ball_diameter + padding) + ball_radius
            color = random.choice(colors)
            cv2.circle(image, (x, y), ball_radius, color, -1, lineType=cv2.LINE_8)

    cv2.imwrite(output_path, image)
    print(f"[✓] Clean 4K ball grid saved to {output_path}")
    return output_path

def add_colored_salt_pepper_noise(image, amount=0.05):
    noisy = image.copy()
    h, w, _ = noisy.shape
    num_noisy = int(amount * h * w)

    ys = np.random.randint(1, h - 1, num_noisy)
    xs = np.random.randint(1, w - 1, num_noisy)

    # for y, x in zip(ys, xs):
    #     color = np.random.choice([0, 255], size=3)
    #     noisy[y-1:y+2, x-1:x+2] = color  # 3x3 block

    for y, x in zip(ys, xs):
        noisy[y, x] = np.random.choice([0, 255], size=3)
    

    return noisy

def image_to_noisy_lossless_video(image_path, output_video_path='noisy_output.mp4', fps=24, duration_sec=2, noise_amount=0.1, temp_dir='input_data/temp_frames'):
    total_frames = fps * duration_sec
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    height, width, _ = image.shape

    # Clean temp dir
    temp_path = Path(temp_dir)
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)

    print(f"[✓] Generating {total_frames} noisy frames...")
    for i in range(total_frames):
        noisy = add_colored_salt_pepper_noise(image, amount=noise_amount)
        frame_path = temp_path / f"frame_{i:04d}.png"
        cv2.imwrite(str(frame_path), noisy)

    # FFmpeg: encode PNG frames to lossless H.264 in MP4
    print("[✓] Encoding video with ffmpeg (lossless H.264)...")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(temp_path / "frame_%04d.png"),
        "-c:v", "libx264",
        "-preset", "veryslow",
        "-crf", "0",
        "-pix_fmt", "yuv444p",
        output_video_path
    ]
    subprocess.run(cmd, check=True)

    print(f"[✓] Lossless video saved to: {output_video_path}")

    # Optional: Clean up frames
    shutil.rmtree(temp_path)

def score_denoise(original_path, denoised_path, max_points = 10):

    original = cv2.imread(original_path)
    denoised = cv2.imread(denoised_path)

    if original.shape != denoised.shape:
        raise ValueError("Images must have the same dimensions and channels.")

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
  
    ssim_value = ssim(original_gray, denoised_gray)

    if ssim_value >= 0.999:
        ratio = 1.0
    elif ssim_value >= 0.990:
        ratio = 0.90
    elif ssim_value >= 0.980:
        ratio = 0.75
    elif ssim_value >= 0.950:
        ratio = 0.50
    else:
        ratio = 0.0

    sub_challenge_points = max_points * ratio

    return sub_challenge_points

def score_ball_counts(true_counts, predicted_counts, max_score=10):
    if len(true_counts) != len(predicted_counts):
        raise ValueError("Predicted and true counts must be the same length")

    total_score = 0
    per_item_scores = []
    
    for true, pred in zip(true_counts, predicted_counts):
        error = abs(true - pred) / true

        if error <= 0.01:
            score = 1.0
        elif error <= 0.05:
            score = 0.5
        elif error <= 0.10:
            score = 0.2
        else:
            score = 0

        per_item_scores.append(score)
        total_score += score

    # Scale to max_score
    final_score = round((total_score / len(true_counts)) * max_score, 2)
    
    return final_score, per_item_scores

if __name__ == "__main__":

    os.makedirs('input_data', exist_ok=True)

    framepath = 'input_data/input_image.png'
    outputpath = 'input_data/output_video.mp4'
    denoised_path = 'input_data/denoised_output.png'
    true_counts = [369, 358, 357, 357, 351, 350, 348, 347, 327, 314]
    

    do_generate = True
    do_solve = True
    do_score = True

    # Generate a clean ball grid image
    
    if do_generate:
        # Generate a clean ball grid image
        framepath = generate_4k_ball_grid(output_path=framepath)
        
        # Add noise to the image and save as a video
        image_to_noisy_lossless_video(framepath, output_video_path=outputpath, fps=24, duration_sec=1, noise_amount=0.2)

    if do_solve:
        denoise_video_to_image(outputpath, output_image_path=denoised_path)

    # Assuming you have a denoised image saved at 'denoised_output.png'
    # Compare the original and denoised images

    if do_score:
        # Score the denoising
        score_sc11 = score_denoise(framepath, denoised_path, max_points=5)
        print('bubble SC 1.1 points:', score_sc11)

    # Segment balls by color

    if do_solve:
        final_list = solve_color_ball_count(denoised_path)
        print("Predicted ball counts:", final_list)

    if do_score:
        #predicted_counts = [400, 373, 366, 351, 348, 341, 335, 334, 312, 308]
        predicted_counts = final_list

        score_sc12, breakdown = score_ball_counts(true_counts, predicted_counts, max_score=15)

        print('bubble SC 1.2 points:', score_sc12, 'Breakdown:', breakdown)

        # Final score
        final_score = score_sc11 + score_sc12
        print(f"Final score: {final_score:.2f} points")




