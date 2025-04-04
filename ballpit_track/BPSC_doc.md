# 🎯 Ball Pit Side Challenge

Welcome to **Ball Pit SC** — a fun side challenge where you can show off your skills in denoising and counting!

---

## 👀 What the Player Sees

You receive:
- A noisy `.mp4` video containing colored balls on a grid.
- Your task has **two parts**:
  1. **Denoise** the video to recover the clean original image.
  2. **Count the number of balls** per color (you don’t need to label the colors — just count and sort them by frequency).

You are **not** told:
- Which colors are used.
- How many different colors are present.
- What the original image looks like.

You must infer everything from the noisy video alone.

Your submission should include:
- A denoised image.
- A list of ball counts (e.g., `[203, 103, 30, ...]`, sorted from most to least).

---

## 🧠 What the Jury Needs to Know

### 🧼 SC 1.1 — Denoising Accuracy (max 5 pts)

Compare the submitted denoised image against the original clean frame using SSIM.

```python
score_denoise(original_path, denoised_path, max_points=5)
```

| SSIM Value      | Points Awarded |
|------------------|----------------|
| ≥ 0.999          | 5.0 pts        |
| ≥ 0.990          | 4.5 pts        |
| ≥ 0.980          | 3.75 pts       |
| ≥ 0.950          | 2.5 pts        |
| < 0.950          | 0 pts          |

---

### 🎨 SC 1.2 — Ball Count Accuracy (max 15 pts)

Compare the submitted count list with the true count list using per-item relative error:

```python
score_ball_counts(true_counts, predicted_counts, max_score=15)
```

| % Error (per color) | Score |
|---------------------|-------|
| ≤ 1%                | 1.0   |
| ≤ 5%                | 0.5   |
| ≤ 10%               | 0.2   |
| > 10%               | 0.0   |

The final score is the **average of all item scores**, scaled to 15 points.

---

### 🧮 Final Score

```python
final_score = score_denoise(...) + score_ball_counts(...)
```

**Total: max 20 points**

---

### ✅ Jury Instructions (How to Run Evaluation)

1. Make sure the following exist:
   - `input_image.png`: original clean frame
   - `output_video.mp4`: the noisy video given to players
   - `denoised_output.png`: player's denoised result
   - `true_counts`: list of correct ball counts
   - `predicted_counts`: player-submitted sorted list

2. Run the test script:

```python
score_sc11 = score_denoise("input_image.png", "denoised_output.png", max_points=5)

score_sc12, breakdown = score_ball_counts(true_counts, predicted_counts, max_score=15)

final_score = score_sc11 + score_sc12
print(f"Final Score: {final_score:.2f} / 20")
```

You can adjust paths or plug this logic into a leaderboard script.

---

## 🧪 What Was Done in the Backend

- A clean synthetic image was generated using flat-colored balls from a fixed palette (10 distinct colors).
- The image was duplicated into 24 frames.
- **Salt-and-pepper colored noise** was added to each frame.
- The noisy frames were compiled into a **lossless H.264 MP4**.
- The original image and true ball counts were stored for evaluation.
- Denoising is expected to be solved via **temporal median filtering**.
- Color segmentation must be solved via **unsupervised clustering + blob detection**.

---

Estimated time to complete: 1-2 hours.
