# 🍌🍉 Flag Finding Track

Welcome to the **Flag Finding Track** — a four-part visual reasoning challenge where you’ll need to **spot hidden flags** using object detection, model fine-tuning and more!

---

## 👀 What the Player Sees

You receive:
- A folder of **5,000 images** (COCO-style visuals).
- A **YOLOv8 model** pre-trained on COCO classes (with no knowledge of dragon fruit).
- A **custom fruit detection dataset**, which includes *dragon fruit*, *snake fruit*, and *pineapple*, with bounding boxes.

Your task has **four parts**:

---

### 🟡 Flag 1: Banana Hunt

- The flag is **hidden inside an image that contains at least one banana**.
- Use the provided YOLOv8 model (COCO-trained) to detect bananas and narrow down the search.

---

### 🔴 Flag 2: Dragon Fruit Mystery

- The flag is **hidden inside an image containing a dragon fruit**.
- Since *dragon fruit* is not in COCO, you must **finetune** the model using the provided fruit dataset.
- Once trained, use your updated model to search the full image set.

---

Your submission must include:
- The **two filenames** where the flags are located.
- The **actual flags** extracted from the images.
- A **clear explanation of your process** — including:
  - Model(s) used
  - Detection output (can be JSON, printed logs, annotated images, etc.)
  - Training configs if you finetuned

> ⚠️ **Submissions without a clear process log or detection evidence may receive significantly fewer points**, even if the correct flags are found.

---

## 🧠 What the Jury Needs to Know

### 🥇 TRACK 2.1 — Banana Flag Hunt (max 10 pts)

- The flag is `G7#xP@2r!M` 
- The image is: `000000459887.jpg`.
```python
score_flag_1 = check_flag(submitted_flag_1, correct_flag="G7#xP@2r!M", max_points=10)
```

| Outcome                        | Points |
|-------------------------------|--------|
| Correct flag + detection process shown | 10 pts |
| Correct flag only, no evidence         | ≤ 4 pts |
| Incorrect flag                         | 0 pts  |

---

### 🥈 TRACK 2.2 — Dragon Fruit Flag Hunt (max 10 pts)


- The flag is `R2#k8@Lm!X` 
- The image is: `000000415741.jpg`.

```python
score_flag_2 = check_flag(submitted_flag_2, correct_flag="R2#k8@Lm!X", max_points=10)
```

| Outcome                        | Points |
|-------------------------------|--------|
| Correct flag + fine-tuning shown     | 10 pts |
| Correct flag only, no training shown | ≤ 4 pts |
| Incorrect flag                       | 0 pts  |

---

### 🧮 Final Score

```python
final_score = score_flag_1 + score_flag_2
```

**Total: max 20 points**

---

## ✅ Jury Instructions (How to Evaluate)

1. Open the submission folder.
2. Confirm:
   - `flag1.txt` and `flag2.txt` contain the extracted strings.
   - The correct filenames are indicated.
   - A folder (or notebook) contains evidence of the detection pipeline.
3. Run:

```python
flag1 = open("flag1.txt").read().strip()
flag2 = open("flag2.txt").read().strip()

score1 = check_flag(flag1, "G7#xP@2r!M", max_points=10)
score2 = check_flag(flag2, "R2#k8@Lm!X", max_points=10)

print(f"Final Score: {score1 + score2} / 20")
```

---

## 🧪 What Was Done in the Backend

- Two flags were hidden in visually complex images.
- The **banana flag** image (`000000459887.jpg`) was included in the original COCO-style set.
- The **dragon fruit flag** image (`000000415741.jpg`) was added **after** the base model was trained, and cannot be found without **finetuning** on the extra dataset.
- Ground truth flag strings are used to evaluate correctness.

Estimated time to complete: **30–60 minutes**, depending on hardware and approach.
