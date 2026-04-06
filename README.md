# 🛡️ Adversarial Shield v3

**Protect your photos from AI facial recognition scraping using adversarial perturbations.**

Built on FGSM (Goodfellow et al., 2014) + PGD (Madry et al., 2017) + FaceNet identity attack.

---

## What's New in v3

| Feature | v1/v2 | v3 |
|---|---|---|
| Attack model | ResNet50 ImageNet | **InceptionResnetV1 (VGGFace2)** |
| Attack target | Object class label | **Face identity embedding (512-d)** |
| Loss function | Cross-entropy | **Negative cosine similarity** |
| Face detection | None (full image) | **MTCNN aligned face crop** |
| Protection metric | Class confidence | **Cosine similarity score** |
| FGSM support | CLI only | **CLI + Web UI toggle** |
| Interface | CLI only | **CLI + Web UI** |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the web UI
python app.py
# Open http://localhost:5000

# OR: use the CLI
python protect.py --image face.jpg --epsilon 0.02
python protect.py --image face.jpg --sweep
python protect.py --image face.jpg --fgsm     # single-step FGSM
```

---

## Project Structure

```
adversarial_shield/
├── app.py              Flask web server
├── face_model.py       FaceNet + MTCNN loader, norm helpers
├── attack.py           PGD + FGSM core, identity loss, epsilon sweep
├── protect.py          CLI pipeline
├── visualize.py        Matplotlib sweep dashboard
├── requirements.txt    Dependencies
└── templates/
    └── index.html      Web UI  ← must be here (Flask requirement)
```

---

## How It Works

### The Real Attack (FaceNet mode)

```
face.jpg
   │
   ▼
MTCNN ──► aligned 160×160 face crop
   │
   ▼
InceptionResnetV1 ──► 512-d embedding vector  [your identity, L2-normalised]
   │
   ▼ PGD: n steps, α = ε/n
   │  Loss = -cosine_similarity(adv_emb, orig_emb)  [minimise = push away]
   │  Each step: pixel_adv -= α · sign(∇_pixel loss)
   │  L∞ projection: keep within ε of original pixels, clamp [0,1]
   │
   ▼
Protected image ──► InceptionResnetV1 ──► different 512-d vector
                                          cosine_sim < 0.5 = ✅ different identity
```

### Why cosine similarity < 0.5 matters

Real face recognition systems store your embedding, then compare new photos
against it using cosine similarity. A threshold of ~0.5 is standard —
below it, the system treats the face as a different (unknown) person.

### FGSM vs PGD

| | FGSM | PGD |
|---|---|---|
| Steps | 1 | n (default 10) |
| Strength | Weaker | Stronger |
| Speed | Faster | Slower |
| When to use | Quick demo, small ε | Real protection |

---

## CLI Usage

```bash
# Single image, default settings (ε=0.02, 10 PGD steps)
python protect.py --image face.jpg

# Stronger protection (higher ε = more noise but stronger attack)
python protect.py --image face.jpg --epsilon 0.05 --steps 20

# Single-step FGSM (faster but weaker)
python protect.py --image face.jpg --fgsm

# Epsilon sweep — compare ε=0.01, 0.05, 0.10 side by side
python protect.py --image face.jpg --sweep

# No visualization (headless/server)
python protect.py --image face.jpg --no-viz
```

---

## Output Metrics

| Metric | Meaning | Target |
|---|---|---|
| Cosine similarity | How similar the protected embedding is to original | < 0.5 |
| Identity distance | 1 - cosine_similarity | > 0.5 |
| PSNR | Visual quality (higher = less visible noise) | > 40 dB |
| Protected | Whether a matcher would recognise you | ✅ |

---

## Choosing Epsilon

| ε | Visual quality | Protection strength | Recommended for |
|---|---|---|---|
| 0.01 | PSNR ~48 dB, invisible | Weak | Testing |
| 0.02 | PSNR ~42 dB, invisible | Moderate | **Default** |
| 0.05 | PSNR ~34 dB, barely visible | Strong | High-risk photos |
| 0.10 | PSNR ~28 dB, slightly visible | Very strong | Max protection |

---

## Honest Scope

✅ **Demonstrated:** Attacks InceptionResnetV1 (VGGFace2) — a real face recognition architecture
✅ **Demonstrated:** Cosine similarity drops below 0.5 at ε ≥ 0.02 (identity hidden from this model)
⚠️ **Partial:** Adversarial transferability — may or may not fool *other* models (ArcFace, DeepFace)
⚠️ **Not guaranteed:** Commercial APIs (Clearview, AWS Rekognition) use proprietary models

For production-grade protection against all scrapers, see: [Fawkes (UChicago)](https://sandlab.cs.uchicago.edu/fawkes/)

---

## References

1. Goodfellow et al. — *Explaining and Harnessing Adversarial Examples* (2014) — FGSM
2. Madry et al. — *Towards Deep Learning Models Resistant to Adversarial Attacks* (2017) — PGD
3. Schroff et al. — *FaceNet: A Unified Embedding for Face Recognition* (2015)
4. Cao et al. — *Fawkes: Protecting Privacy against Facial Recognition* (USENIX 2020)
