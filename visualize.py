"""
visualize.py — Visualizations for FaceNet identity attack and ResNet50 fallback.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image
from attack import compute_psnr


def _to_np(tensor: torch.Tensor, denorm) -> np.ndarray:
    t = denorm(tensor).squeeze(0).cpu().detach().clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _pil_np(pil_img, size=None):
    if size:
        pil_img = pil_img.resize(size, Image.LANCZOS)
    return np.array(pil_img)


def _style(ax, color):
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_facecolor("#0d1117")
    for s in ax.spines.values():
        s.set_edgecolor(color)
        s.set_linewidth(2)


def show_comparison(face_pil, pil_protected, metrics, epsilon, model_type,
                    save_path="comparison.png"):
    target_size = (160, 160) if model_type == 'facenet' else (224, 224)
    orig_np  = _pil_np(face_pil, target_size)
    prot_np  = _pil_np(pil_protected, target_size)
    noise_np = np.clip(
        (prot_np.astype(int) - orig_np.astype(int)) * 10 + 128, 0, 255
    ).astype(np.uint8)
    psnr = compute_psnr(orig_np, prot_np)

    if model_type == 'facenet':
        sim        = metrics['cosine_similarity']
        prot       = metrics['protected']
        orig_label = "Original Face\n(identity: you)"
        adv_label  = (f"Cosine sim: {sim:.3f}\n"
                      f"{'✅ Identity hidden' if prot else '⚠️ Try higher ε'}")
    else:
        orig_label = (f"{metrics.get('orig_class_name', metrics['orig_class'])}\n"
                      f"{metrics['orig_confidence']:.1%}")
        adv_label  = (f"{metrics.get('adv_class_name', metrics['adv_class'])}\n"
                      f"{metrics['adv_confidence']:.1%}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor("#0f0f14")

    for ax, img, title, label, color in zip(
        axes,
        [orig_np, noise_np, prot_np],
        ["Original", "Noise ×10", "Protected"],
        [orig_label, f"PSNR = {psnr:.1f} dB\n(>40 dB = invisible)", adv_label],
        ["#4ade80", "#facc15", "#f87171"],
    ):
        ax.imshow(img)
        ax.set_title(title, color="#e2e8f0", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(label, color=color, fontsize=10, labelpad=6)
        _style(ax, color)

    attack_name = ("FaceNet Identity Attack" if model_type == 'facenet'
                   else "ResNet50 Classifier Attack")
    fig.suptitle(f"Adversarial Shield — {attack_name}   (ε={epsilon})",
                 color="#fff", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[viz] Saved → {save_path}")
    plt.show()


def show_sweep_dashboard(results, face_pil, denorm, model_type, class_names=None,
                         save_path="epsilon_sweep.png"):
    n   = len(results)
    fig = plt.figure(figsize=(4 * (n + 1), 10))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, n + 1, figure=fig, hspace=0.55, wspace=0.15,
                            height_ratios=[3, 2])

    target_size = (160, 160) if model_type == 'facenet' else (224, 224)
    orig_np = _pil_np(face_pil, target_size)

    # ── Row 0: images ──────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(orig_np)
    ax0.set_title("Original", color="#e2e8f0", fontsize=10, fontweight="bold")
    ax0.set_xlabel("True identity", color="#4ade80", fontsize=8)
    _style(ax0, "#4ade80")

    for i, r in enumerate(results):
        ax     = fig.add_subplot(gs[0, i + 1])
        adv_np = _to_np(r['perturbed_tensor'], denorm)
        psnr   = compute_psnr(orig_np, adv_np)
        ax.imshow(adv_np)
        ax.set_title(f"ε={r['epsilon']:.2f}  |  PSNR {psnr:.1f}dB",
                     color="#e2e8f0", fontsize=9, fontweight="bold")

        if model_type == 'facenet':
            sim  = r['cosine_similarity']
            prot = "✅ Hidden" if r['protected'] else "⚠️ Partial"
            ax.set_xlabel(f"Sim: {sim:.3f} — {prot}", color="#f87171", fontsize=8)
        else:
            cls  = class_names[r['adv_class']] if class_names else str(r['adv_class'])
            ax.set_xlabel(f"{cls[:20]}\n{r['adv_confidence']:.1%}",
                          color="#f87171", fontsize=8)
        _style(ax, "#f87171")

    # ── Row 1: metric bar chart ────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, :])
    epsilons = [r['epsilon'] for r in results]
    x     = np.arange(len(epsilons))
    width = 0.35

    if model_type == 'facenet':
        orig_vals  = [1.0] * len(results)
        adv_vals   = [r['cosine_similarity'] for r in results]
        ylabel     = "Cosine Similarity (1.0 = same identity)"
        ylim       = (0, 1.15)
        orig_label = "Original similarity"
        adv_label  = "After protection"
    else:
        orig_vals  = [r['orig_confidence'] * 100 for r in results]
        adv_vals   = [r['adv_confidence']  * 100 for r in results]
        ylabel     = "Model Confidence (%)"
        ylim       = (0, 115)
        orig_label = "Original"
        adv_label  = "Adversarial"

    b1 = ax_bar.bar(x - width / 2, orig_vals, width,
                    label=orig_label, color="#4ade80", alpha=0.85, zorder=3)
    b2 = ax_bar.bar(x + width / 2, adv_vals,  width,
                    label=adv_label,  color="#f87171", alpha=0.85, zorder=3)

    for bar, color in [*[(b, "#4ade80") for b in b1], *[(b, "#f87171") for b in b2]]:
        h = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            h + (0.02 if model_type == 'facenet' else 1),
            f"{h:.2f}" if model_type == 'facenet' else f"{h:.0f}%",
            ha="center", va="bottom", color=color, fontsize=8, fontweight="bold",
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"ε={e:.2f}" for e in epsilons],
                           color="#cbd5e1", fontsize=10)
    ax_bar.set_ylabel(ylabel, color="#94a3b8", fontsize=9)
    ax_bar.set_ylim(*ylim)
    ax_bar.set_facecolor("#161b22")
    ax_bar.tick_params(colors="#94a3b8")
    ax_bar.grid(axis="y", color="#334155", linestyle="--", alpha=0.5, zorder=0)
    for s in ax_bar.spines.values():
        s.set_edgecolor("#334155")
    ax_bar.legend(facecolor="#1e293b", edgecolor="#475569",
                  labelcolor="#e2e8f0", fontsize=9)

    if model_type == 'facenet':
        ax_bar.axhline(0.5, color="#facc15", linestyle="--", linewidth=1.5, zorder=4)
        ax_bar.text(len(epsilons) - 0.5, 0.52, "Identity threshold (0.5)",
                    color="#facc15", fontsize=8)

    attack_name = ("FaceNet Identity Attack" if model_type == 'facenet'
                   else "ResNet50 Classifier Attack")
    fig.suptitle(f"Adversarial Shield — {attack_name} — Epsilon Sweep",
                 color="#fff", fontsize=13, fontweight="bold", y=1.01)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[viz] Saved → {save_path}")
    plt.show()
