"""
protect.py — CLI: load → detect face → attack identity → save → report.

Usage
─────
  python protect.py --image face.jpg                   # default ε=0.02, PGD
  python protect.py --image face.jpg --epsilon 0.05
  python protect.py --image face.jpg --sweep           # ε ∈ {0.01, 0.05, 0.10}
  python protect.py --image face.jpg --steps 20        # more PGD iterations
  python protect.py --image face.jpg --fgsm            # single-step FGSM
  python protect.py --image face.jpg --no-viz          # skip matplotlib
"""

import argparse
import os
import sys
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from face_model import (load_face_model, load_detector, preprocess_for_facenet,
                        get_denorm_renorm, load_class_labels, IMAGENET_MEAN, IMAGENET_STD)
from attack     import protect_image, epsilon_sweep, compute_psnr
from visualize  import show_comparison, show_sweep_dashboard


def tensor_to_pil(tensor: torch.Tensor, denorm) -> Image.Image:
    t = denorm(tensor).squeeze(0).cpu().detach().clamp(0, 1)
    return TF.to_pil_image(t)


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_banner(model_type):
    mode = ("FaceNet (VGGFace2) — REAL identity attack" if model_type == 'facenet'
            else "ResNet50 (ImageNet) — classifier attack [fallback]")
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          Adversarial Shield v3 — Identity Protector          ║
║  FGSM (2014) · PGD (2017) · Fawkes-inspired (2020)          ║
║  Model: {mode[:52]:52s}║
╚══════════════════════════════════════════════════════════════╝
""")


def protect(args):
    device             = pick_device()
    model, model_type  = load_face_model(device)
    detector           = load_detector(device)
    denorm, renorm     = get_denorm_renorm(model_type)

    print_banner(model_type)
    print(f"[1/5] Device      : {device}")
    print(f"[2/5] Model       : {'InceptionResnetV1 VGGFace2' if model_type == 'facenet' else 'ResNet50 ImageNet'}")

    if not os.path.exists(args.image):
        sys.exit(f"Error: file not found — {args.image}")

    print(f"[3/5] Loading     : {args.image}")
    pil_original                        = Image.open(args.image).convert("RGB")
    image_tensor, face_pil, face_detected = preprocess_for_facenet(
        pil_original, detector, device
    )

    if face_detected:
        print(f"      Face detected ✓ — attacking aligned face crop (160×160)")
    else:
        print(f"      No face detected — using full image")

    target_label = None
    class_names  = None
    if model_type == 'resnet50':
        class_names = load_class_labels()
        with torch.no_grad():
            logits = model(image_tensor)
        target_label = logits.argmax(dim=1).item()
        print(f"      Top-1 class: {class_names[target_label]} "
              f"({torch.softmax(logits, 1).max().item():.1%})")

    attack_label = "FGSM (single step)" if args.fgsm else f"PGD steps={args.steps}"

    # ── Sweep ──────────────────────────────────────────────────────────
    if args.sweep:
        epsilons = [0.01, 0.05, 0.10]
        print(f"\n[4/5] Epsilon sweep ε ∈ {epsilons}, {attack_label}\n")

        results = epsilon_sweep(
            model, model_type, image_tensor,
            denorm, renorm, epsilons, device,
            n_steps=args.steps,
            target_label=target_label,
            use_fgsm=args.fgsm,
        )

        _print_sweep_table(results, model_type, class_names, denorm, face_pil)

        print("\n[5/5] Saving sweep dashboard → epsilon_sweep.png")
        if not args.no_viz:
            show_sweep_dashboard(results, face_pil, denorm, model_type, class_names)
        return

    # ── Single epsilon ──────────────────────────────────────────────────
    print(f"[4/5] Attacking    : ε = {args.epsilon}, {attack_label}")
    perturbed, metrics = protect_image(
        model, model_type, image_tensor,
        denorm, renorm, args.epsilon, args.steps, device,
        target_label=target_label,
        use_fgsm=args.fgsm,
    )

    pil_protected = tensor_to_pil(perturbed, denorm)
    pil_protected.save(args.output)
    print(f"[5/5] Saved        : {args.output}")

    _print_results(metrics, model_type, class_names)

    if not args.no_viz:
        show_comparison(face_pil, pil_protected, metrics, args.epsilon, model_type)


def _print_results(metrics, model_type, class_names):
    print()
    print("  ┌────────────────────────────────────────────────────┐")
    print("  │  RESULTS                                           │")
    if model_type == 'facenet':
        sim  = metrics['cosine_similarity']
        dist = metrics['identity_distance']
        prot = "✅ YES" if metrics['protected'] else "⚠️  NO (try higher ε)"
        print(f"  │  Cosine similarity  : {sim:.4f}  (1.0 = same person)   │")
        print(f"  │  Identity distance  : {dist:.4f}  (>0.5 = different ID) │")
        print(f"  │  Protected          : {prot:40s}│")
    else:
        orig = class_names[metrics['orig_class']] if class_names else str(metrics['orig_class'])
        adv  = class_names[metrics['adv_class']]  if class_names else str(metrics['adv_class'])
        print(f"  │  Original class     : {orig[:28]:28s} {metrics['orig_confidence']:.1%}  │")
        print(f"  │  Adversarial class  : {adv[:28]:28s} {metrics['adv_confidence']:.1%}  │")
        prot = "✅ YES" if metrics['protected'] else "⚠️  NO"
        print(f"  │  Misclassified      : {prot}                              │")
    print("  └────────────────────────────────────────────────────┘")


def _print_sweep_table(results, model_type, class_names, denorm, face_pil):
    import numpy as np
    orig_np = np.array(face_pil.resize((160, 160)))

    if model_type == 'facenet':
        print(f"  {'ε':>6}  {'Similarity':>12}  {'Distance':>10}  {'Protected':>10}  {'PSNR':>8}")
        print("  " + "─" * 60)
        for r in results:
            adv_np = (denorm(r['perturbed_tensor']).squeeze(0).cpu().detach()
                      .clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
            psnr   = compute_psnr(orig_np, adv_np)
            prot   = "✅" if r['protected'] else "⚠️"
            print(f"  {r['epsilon']:>6.2f}  "
                  f"{r['cosine_similarity']:>12.4f}  "
                  f"{r['identity_distance']:>10.4f}  "
                  f"{prot:>10}  "
                  f"{psnr:>7.1f}dB")
    else:
        print(f"  {'ε':>6}  {'Orig class':>22}  {'Orig conf':>10}  "
              f"{'Adv class':>22}  {'Adv conf':>9}")
        print("  " + "─" * 80)
        for r in results:
            print(f"  {r['epsilon']:>6.2f}  "
                  f"{class_names[r['orig_class']]:>22}  "
                  f"{r['orig_confidence']:>9.1%}  "
                  f"{class_names[r['adv_class']]:>22}  "
                  f"{r['adv_confidence']:>8.1%}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Protect face from AI scraping via adversarial perturbation."
    )
    p.add_argument("--image",   required=True,              help="Path to input image")
    p.add_argument("--epsilon", type=float, default=0.02,   help="L∞ perturbation budget (default 0.02)")
    p.add_argument("--output",  default="protected.png",    help="Output path (default: protected.png)")
    p.add_argument("--sweep",   action="store_true",        help="Sweep ε ∈ {0.01, 0.05, 0.10}")
    p.add_argument("--steps",   type=int, default=10,       help="PGD iterations (default 10)")
    p.add_argument("--fgsm",    action="store_true",        help="Use single-step FGSM instead of PGD")
    p.add_argument("--no-viz",  action="store_true",        help="Skip matplotlib visualisation")
    return p.parse_args()


if __name__ == "__main__":
    protect(parse_args())
