"""
attack.py — Adversarial identity attack: FaceNet MI-FGSM + ResNet50 fallback.

Sign convention (correct)
──────────────────────────
  loss = cosine_similarity(adv_emb, orig_emb)   [positive, we minimise it]
  pixel_adv -= alpha * sign(grad_loss)           [descent = reduces sim ✓]

Full-image protection
──────────────────────
Attack runs on the 160x160 FaceNet crop.
Pixel delta is upscaled and applied ONLY inside the face bounding box.

Multi-model verification
──────────────────────────
verify_against_models() receives BOTH the original face tensor AND the
already-perturbed tensor, and tests transferability across 5 surrogate models.
ResNet models are used as deep feature extractors (penultimate layer cosine
similarity) — a much more meaningful transfer metric than class labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Callable, Optional


# ── Identity loss (correct sign) ─────────────────────────────────────────

class IdentityLoss(nn.Module):
    """
    loss = cosine_similarity(adv_emb, orig_emb) + MSE(adv_emb, -orig_emb)

    Minimising with gradient descent:
      - cosine term pushes angular distance up
      - MSE term pulls toward antipodal point (prevents gradient plateau)
    Both positive → descent on loss = reduction in similarity ✓
    """
    def __init__(self, orig_embedding: torch.Tensor):
        super().__init__()
        self.orig_emb = orig_embedding.detach()
        self.target   = -F.normalize(orig_embedding, dim=1).detach()

    def forward(self, model_output: torch.Tensor) -> torch.Tensor:
        cos_loss = F.cosine_similarity(model_output, self.orig_emb, dim=1).mean()
        mse_loss = F.mse_loss(model_output, self.target)
        return cos_loss + mse_loss


# ── MI-FGSM ───────────────────────────────────────────────────────────────

def mi_fgsm_attack(model, image_tensor, loss_fn, denorm, renorm,
                   epsilon, n_steps, device, decay=1.0):
    """Momentum Iterative FGSM — gradient DESCENT on loss_fn."""
    alpha      = epsilon / n_steps
    pixel_orig = denorm(image_tensor).detach()
    noise      = torch.empty_like(pixel_orig).uniform_(-epsilon * 0.5, epsilon * 0.5)
    pixel_adv  = (pixel_orig + noise).clamp(0, 1)
    momentum   = torch.zeros_like(pixel_orig)

    for _ in range(n_steps):
        pixel_adv = pixel_adv.detach().requires_grad_(True)
        if hasattr(model, 'zero_grad'):
            model.zero_grad()
        output = model(renorm(pixel_adv))
        loss   = loss_fn(output)
        loss.backward()

        with torch.no_grad():
            grad = pixel_adv.grad.data
            gn       = grad.abs().mean(dim=[1,2,3], keepdim=True).clamp(min=1e-12)
            grad     = grad / gn
            momentum = decay * momentum + grad
            # Gradient DESCENT: subtract sign to minimise loss
            pixel_adv = pixel_adv - alpha * momentum.sign()
            delta     = (pixel_adv - pixel_orig).clamp(-epsilon, epsilon)
            pixel_adv = (pixel_orig + delta).clamp(0, 1)

    return renorm(pixel_adv).detach()


# ── FGSM ──────────────────────────────────────────────────────────────────

def fgsm_attack(model, image_tensor, loss_fn, denorm, renorm, epsilon, device):
    """Single-step FGSM — gradient descent on loss_fn."""
    pixel_orig = denorm(image_tensor).detach()
    pixel_adv  = pixel_orig.clone().requires_grad_(True)
    if hasattr(model, 'zero_grad'):
        model.zero_grad()
    loss = loss_fn(model(renorm(pixel_adv)))
    loss.backward()
    with torch.no_grad():
        pixel_adv = pixel_orig - epsilon * pixel_adv.grad.data.sign()
        pixel_adv = pixel_adv.clamp(0, 1)
    return renorm(pixel_adv).detach()


# ── Core protect_image ────────────────────────────────────────────────────

def protect_image(model, model_type, image_tensor, denorm, renorm,
                  epsilon, n_steps, device,
                  target_label=None, use_fgsm=False):
    """Attack image_tensor (model input space). Returns (perturbed, metrics)."""
    if model_type == 'facenet':
        with torch.no_grad():
            orig_emb = model(image_tensor)   # FaceNet output is already L2-normalised

        loss_fn = IdentityLoss(orig_emb)

        if use_fgsm:
            perturbed = fgsm_attack(model, image_tensor, loss_fn,
                                    denorm, renorm, epsilon, device)
        else:
            perturbed = mi_fgsm_attack(model, image_tensor, loss_fn,
                                       denorm, renorm, epsilon, n_steps, device)

        with torch.no_grad():
            adv_emb = model(perturbed)
            sim     = F.cosine_similarity(orig_emb, adv_emb, dim=1).item()

        metrics = {
            "mode":              "FaceNet Identity Attack (MI-FGSM)",
            "cosine_similarity": sim,
            "identity_distance": 1.0 - sim,
            "protected":         sim < 0.5,
            "orig_embedding_norm": float(orig_emb.norm().item()),
        }

    else:
        criterion    = nn.CrossEntropyLoss()
        label_tensor = torch.tensor([target_label], dtype=torch.long).to(device)
        with torch.no_grad():
            out       = model(image_tensor)
            orig_conf = torch.softmax(out, dim=1).max().item()
            orig_cls  = out.argmax(dim=1).item()

        loss_fn = lambda o: criterion(o, label_tensor)
        if use_fgsm:
            perturbed = fgsm_attack(model, image_tensor, loss_fn,
                                    denorm, renorm, epsilon, device)
        else:
            perturbed = mi_fgsm_attack(model, image_tensor, loss_fn,
                                       denorm, renorm, epsilon, n_steps, device)

        with torch.no_grad():
            adv_out  = model(perturbed)
            adv_conf = torch.softmax(adv_out, dim=1).max().item()
            adv_cls  = adv_out.argmax(dim=1).item()

        metrics = {
            "mode":            "ResNet50 Classifier Attack (fallback)",
            "orig_class":      orig_cls,
            "orig_confidence": orig_conf,
            "adv_class":       adv_cls,
            "adv_confidence":  adv_conf,
            "protected":       adv_cls != orig_cls,
        }

    return perturbed, metrics


# ── Face-only full-image protection ───────────────────────────────────────

def protect_full_image(model, model_type, pil_original, denorm, renorm,
                       epsilon, n_steps, device,
                       target_label=None, use_fgsm=False,
                       face_tensor=None, face_box=None):
    """
    Attack 160x160 face crop, upscale delta, apply ONLY within face_box.
    Returns (protected_pil, face_attacked_pil, perturbed_face_tensor, metrics).
    perturbed_face_tensor is returned so verify_against_models can use it.
    """
    import torchvision.transforms as T

    orig_w, orig_h = pil_original.size

    perturbed_tensor, metrics = protect_image(
        model, model_type, face_tensor, denorm, renorm,
        epsilon, n_steps, device,
        target_label=target_label, use_fgsm=use_fgsm,
    )

    delta_small = denorm(perturbed_tensor) - denorm(face_tensor.detach())
    orig_tensor = T.ToTensor()(pil_original).unsqueeze(0)

    if face_box is not None and model_type == 'facenet':
        x1, y1, x2, y2 = [int(v) for v in face_box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        bw, bh = x2 - x1, y2 - y1

        if bw > 0 and bh > 0:
            delta_box = F.interpolate(
                delta_small.cpu().float(), size=(bh, bw),
                mode='bilinear', align_corners=False
            )
            protected_tensor = orig_tensor.clone()
            protected_tensor[:, :, y1:y2, x1:x2] = (
                orig_tensor[:, :, y1:y2, x1:x2] + delta_box
            ).clamp(0, 1)
        else:
            protected_tensor = orig_tensor.clone()
    else:
        delta_full = F.interpolate(
            delta_small.cpu().float(), size=(orig_h, orig_w),
            mode='bilinear', align_corners=False
        )
        protected_tensor = (orig_tensor + delta_full).clamp(0, 1)

    protected_pil = T.ToPILImage()(protected_tensor.squeeze(0))
    face_disp     = denorm(perturbed_tensor).squeeze(0).cpu().detach().clamp(0, 1)
    face_pil      = T.ToPILImage()(face_disp)

    # Return perturbed_tensor so caller can pass it to verify_against_models
    return protected_pil, face_pil, perturbed_tensor, metrics


# ── Multi-model verification (FIXED) ─────────────────────────────────────

def _resnet_embed(model, face_tensor_01, device):
    """
    Extract penultimate-layer features from a ResNet/MobileNet as a
    face embedding proxy. Much more meaningful than class labels.
    face_tensor_01: (1,3,160,160) in [0,1] range.
    """
    # Resize to 224x224 and apply ImageNet normalisation
    face_224 = F.interpolate(face_tensor_01, size=(224, 224),
                              mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    face_norm = (face_224 - mean) / std

    # Hook penultimate layer (before final classifier)
    features = {}
    def hook(m, i, o):
        features['feat'] = o.detach()

    # ResNet: avgpool output; MobileNet: classifier[0] input
    if hasattr(model, 'avgpool'):
        h = model.avgpool.register_forward_hook(hook)
    else:
        h = model.features.register_forward_hook(hook)

    with torch.no_grad():
        model(face_norm)
    h.remove()

    feat = features['feat'].flatten(1)
    return F.normalize(feat, dim=1)


def verify_against_models(orig_face_tensor, perturbed_face_tensor, device):
    """
    Test adversarial transferability across 5 surrogate models.

    orig_face_tensor      : (1,3,160,160) in FaceNet [-1,1] range — ORIGINAL
    perturbed_face_tensor : (1,3,160,160) in FaceNet [-1,1] range — AFTER ATTACK

    For each model:
      - Extract deep features from ORIGINAL face
      - Extract deep features from PERTURBED face
      - Compute cosine similarity between them
      - sim < 0.5 → protected (embeddings are far apart)

    Returns list of result dicts.
    """
    import torchvision.models as tvm

    # Convert from FaceNet [-1,1] to [0,1] for surrogate models
    orig_01 = ((orig_face_tensor + 1) / 2).clamp(0, 1).to(device)
    adv_01  = ((perturbed_face_tensor + 1) / 2).clamp(0, 1).to(device)

    results = []

    # ── Model 1: FaceNet VGGFace2 (white-box — same as attack model) ──────
    try:
        from facenet_pytorch import InceptionResnetV1
        m1 = InceptionResnetV1(pretrained='vggface2').to(device).eval()
        with torch.no_grad():
            # Convert [0,1] back to FaceNet [-1,1]
            oe = m1(orig_01 * 2 - 1)
            ae = m1(adv_01  * 2 - 1)
        # Catch NaN (happens if weights failed to download)
        if torch.isnan(oe).any() or torch.isnan(ae).any():
            raise ValueError("NaN in embedding — model weights unavailable in this environment")
        sim = F.cosine_similarity(oe, ae, dim=1).item()
        results.append({
            "model": "FaceNet VGGFace2 (white-box)",
            "cosine_similarity": sim,
            "protected": sim < 0.5,
        })
    except Exception as e:
        results.append({"model": "FaceNet VGGFace2 (white-box)", "error": str(e)[:80]})

    # ── Model 2: FaceNet CASIA-WebFace (transfer — different training set) ─
    try:
        from facenet_pytorch import InceptionResnetV1
        m2 = InceptionResnetV1(pretrained='casia-webface').to(device).eval()
        with torch.no_grad():
            oe = m2(orig_01 * 2 - 1)
            ae = m2(adv_01  * 2 - 1)
        if torch.isnan(oe).any() or torch.isnan(ae).any():
            raise ValueError("NaN in embedding — model weights unavailable")
        sim = F.cosine_similarity(oe, ae, dim=1).item()
        results.append({
            "model": "FaceNet CASIA-WebFace (transfer)",
            "cosine_similarity": sim,
            "protected": sim < 0.5,
        })
    except Exception as e:
        results.append({"model": "FaceNet CASIA-WebFace (transfer)", "error": str(e)[:80]})

    # ── Models 3-5: ResNet/MobileNet as deep feature extractors ───────────
    surrogate_models = [
        ("ResNet50 (features)",   lambda: tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)),
        ("ResNet101 (features)",  lambda: tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V1)),
        ("MobileNetV3 (features)",lambda: tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1)),
    ]

    for name, model_fn in surrogate_models:
        try:
            m = model_fn().to(device).eval()
            orig_feat = _resnet_embed(m, orig_01, device)
            adv_feat  = _resnet_embed(m, adv_01,  device)
            sim = F.cosine_similarity(orig_feat, adv_feat, dim=1).item()
            results.append({
                "model":            name,
                "cosine_similarity": sim,
                "protected":        sim < 0.85,   # deep features are more similar, lower threshold
                "note":             "deep feature similarity (lower = more disrupted)",
            })
        except Exception as e:
            results.append({"model": name, "error": str(e)[:80]})

    return results


# ── Epsilon sweep ─────────────────────────────────────────────────────────

def epsilon_sweep(model, model_type, image_tensor, denorm, renorm,
                  epsilons, device, n_steps=10, target_label=None, use_fgsm=False):
    results = []
    for eps in epsilons:
        img = image_tensor.detach().clone().to(device)
        perturbed, metrics = protect_image(
            model, model_type, img, denorm, renorm, eps, n_steps, device,
            target_label=target_label, use_fgsm=use_fgsm,
        )
        results.append({"epsilon": eps, "perturbed_tensor": perturbed, **metrics})
    return results


# ── PSNR ──────────────────────────────────────────────────────────────────

def compute_psnr(orig_np, adv_np):
    mse = np.mean((orig_np.astype(float) - adv_np.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))
