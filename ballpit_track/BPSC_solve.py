import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

def denoise_video_to_image(video_path, output_image_path='denoised.png'):
    cap = cv2.VideoCapture(video_path)
    frames = []

    print("[✓] Reading video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("No frames were read from the video.")

    print(f"[✓] {len(frames)} frames read. Applying median filtering...")

    # Stack into numpy array (T, H, W, C)
    video_array = np.stack(frames, axis=0)

    # Median across time axis (T)
    denoised = np.median(video_array, axis=0).astype(np.uint8)

    cv2.imwrite(output_image_path, denoised)
    print(f"[✓] Denoised image saved to: {output_image_path}")
    return denoised

def solve_color_ball_count(image_path, max_k=15, subsample_fraction=0.1):
    image = cv2.imread(image_path)

    image = cv2.medianBlur(image, 5)

    h, w, _ = image.shape
    img_flat = image.reshape(-1, 3)

    # 1. Automatically detect background color from top-left corner
    corners = [
        tuple(image[0, 0]), tuple(image[0, -1]),
        tuple(image[-1, 0]), tuple(image[-1, -1])
    ]
    background_color = max(set(corners), key=corners.count)
    print(f"[✓] Detected background color: {background_color}")

    # 2. Remove background pixels
    mask = ~np.all(img_flat == background_color, axis=1)
    pixels = img_flat[mask]

    # 3. Estimate number of unique colors (subsample for speed)
    sample_size = int(len(pixels) * subsample_fraction)
    sample_pixels = pixels[np.random.choice(len(pixels), sample_size, replace=False)]
    unique_colors = np.unique(sample_pixels, axis=0)
    num_clusters = min(len(unique_colors), max_k)
    print(f"[✓] Estimated number of clusters: {num_clusters}")

    # 4. KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    cluster_centers = np.uint8(kmeans.cluster_centers_)

    # 5. Map labels back to full image
    full_labels = np.full((h * w,), -1)
    full_labels[mask] = labels
    label_img = full_labels.reshape((h, w))

    color_counts = []

    for i, center_color in enumerate(cluster_centers):
        mask = (label_img == i).astype(np.uint8) * 255

        # Count blobs (connected components)
        num_labels, _ = cv2.connectedComponents(mask)
        count = num_labels - 1

        color_counts.append((count, tuple(center_color)))

        os.makedirs("segmentation_masks", exist_ok=True)

        # save segmentation map
        cv2.imwrite(f"segmentation_masks/segment_{i+1}.png", mask)

    color_counts.sort(reverse=True)
    print("\n[✓] Ball counts (most to least):")
    for idx, (count, color) in enumerate(color_counts):
        print(f"  {idx+1}. {count} balls  [color: {color}]")

    return [count for count, _ in color_counts]

if __name__ == "__main__":
    video_path = 'output_video.mp4'  # Path to the noisy video
    output_image_path = 'denoised_output.png'  # Path to save the denoised image

    # Denoise video to image
    denoised_image = denoise_video_to_image(video_path, output_image_path)

    # Segment balls by color
    final_list = solve_color_ball_count(output_image_path)