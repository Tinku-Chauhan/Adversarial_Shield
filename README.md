# 🛡️ Adversarial Shield v3

**Protect your photos from AI facial recognition scraping using adversarial perturbations.**

Built on FGSM (Goodfellow et al., 2014) + MI-FGSM (Dong et al., 2018) + FaceNet identity attack.

---

## What's New in v3

| Feature | v1/v2 | v3 |
|---|---|---|
| Attack model | ResNet50 ImageNet | **InceptionResnetV1 (VGGFace2)** |
| Attack algorithm | FGSM / PGD | **MI-FGSM (momentum iterative)** |
| Attack target | Object class label | **Face identity embedding (512-d)** |
| Loss function | Cross-entropy | **Cosine similarity + antipodal MSE** |
| Face detection | None (full image) | **MTCNN aligned face crop** |
| Perturbation scope | Whole image | **Face bounding box only** |
| Protection metric | Class confidence | **Cosine similarity score** |
| FGSM support | CLI only | **CLI + Web UI toggle** |
| Multi-model verify | None | **5 surrogate models** |
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
├── attack.py           MI-FGSM + FGSM core, identity loss, epsilon sweep, verification
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
MTCNN ──► aligned 160×160 face crop  +  bounding box (x1,y1,x2,y2)
   │
   ▼
InceptionResnetV1 ──► 512-d embedding vector  [your identity, L2-normalised]
   │
   ▼ MI-FGSM: n steps, α = ε/n, momentum decay = 1.0
   │  Loss = cosine_similarity(adv_emb, orig_emb)          [minimise → push away]
   │        + MSE(adv_emb, -normalize(orig_emb))           [antipodal anchor]
   │  Each step:
   │    grad  = normalize(∇_pixel loss)                    [unit gradient]
   │    m     = decay·m + grad                             [momentum accumulation]
   │    pixel -= α · sign(m)                               [descent step]
   │    delta = clamp(pixel - orig, -ε, ε)                 [L∞ projection]
   │    pixel = clamp(orig + delta, 0, 1)
   │
   ▼  delta upscaled → applied ONLY inside face bounding box on full image
   │
   ▼
Protected image ──► InceptionResnetV1 ──► different 512-d vector
                                          cosine_sim < 0.5 = ✅ different identity
```

### Identity Loss Design

The loss has two terms working together:

| Term | Formula | Effect |
|---|---|---|
| Cosine | `cos_sim(adv_emb, orig_emb)` | Pushes angular distance apart |
| Antipodal MSE | `MSE(adv_emb, -normalize(orig_emb))` | Pulls toward opposite point, prevents gradient plateau |

Both are positive scalars — gradient descent on their sum reduces identity similarity consistently.

### Why cosine similarity < 0.5 matters

Real face recognition systems store your embedding, then compare new photos against it using cosine similarity. A threshold of ~0.5 is standard — below it, the system treats the face as a different (unknown) person.

### FGSM vs MI-FGSM

| | FGSM | MI-FGSM |
|---|---|---|
| Steps | 1 | n (default 10) |
| Momentum | No | Yes (decay=1.0) |
| Strength | Weaker | **Stronger + better transfer** |
| Speed | Faster | Slower |
| When to use | Quick demo, small ε | **Real protection** |

MI-FGSM uses accumulated momentum across steps, which stabilises gradient directions and significantly improves adversarial transferability to other models compared to vanilla PGD.

### Face-Box Perturbation

The perturbation delta is computed on the 160×160 FaceNet crop, then bilinearly upscaled and composited **only within the MTCNN face bounding box** on the full-resolution image. Background pixels are untouched. If no face is detected, the delta is upscaled to the full image.

---

## Multi-Model Verification

When verification is enabled (Web UI checkbox / always available in CLI), the protected face is tested against 5 surrogate models:

| # | Model | Type | Threshold |
|---|---|---|---|
| 1 | FaceNet VGGFace2 | White-box (same as attack) | sim < 0.5 |
| 2 | FaceNet CASIA-WebFace | Transfer (different training set) | sim < 0.5 |
| 3 | ResNet50 (deep features) | Transfer (penultimate layer cosine) | sim < 0.85 |
| 4 | ResNet101 (deep features) | Transfer (penultimate layer cosine) | sim < 0.85 |
| 5 | MobileNetV3-Large (deep features) | Transfer (penultimate layer cosine) | sim < 0.85 |

ResNet/MobileNet models are used as deep feature extractors (penultimate `avgpool` layer), not classifiers — this is a meaningful transfer proxy for how other recognition systems perceive the face.

---

## CLI Usage

```bash
# Single image, default settings (ε=0.02, 10 MI-FGSM steps)
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
✅ **Demonstrated:** MI-FGSM improves transfer vs vanilla PGD (momentum stabilises gradient direction)  
✅ **Demonstrated:** Multi-model verification across 5 surrogates including CASIA-WebFace FaceNet  
⚠️ **Partial:** Adversarial transferability — may or may not fool *other* models (ArcFace, DeepFace)  
⚠️ **Not guaranteed:** Commercial APIs (Clearview, AWS Rekognition) use proprietary models  

For production-grade protection against all scrapers, see: [Fawkes (UChicago)](https://sandlab.cs.uchicago.edu/fawkes/)

---

## References

1. Goodfellow et al. — *Explaining and Harnessing Adversarial Examples* (2014) — FGSM  
2. Dong et al. — *Boosting Adversarial Attacks with Momentum* (CVPR 2018) — MI-FGSM  
3. Madry et al. — *Towards Deep Learning Models Resistant to Adversarial Attacks* (2017) — PGD  
4. Schroff et al. — *FaceNet: A Unified Embedding for Face Recognition* (2015)  
5. Cao et al. — *Fawkes: Protecting Privacy against Facial Recognition* (USENIX 2020)  
