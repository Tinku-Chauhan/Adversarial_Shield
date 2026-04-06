"""
app.py — Flask web interface for Adversarial Shield.
Run:  python app.py  →  http://localhost:5000
"""

import os, io, base64, torch, numpy as np
from PIL import Image
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file, render_template

from face_model import (load_face_model, load_detector, preprocess_for_facenet,
                        get_denorm_renorm, load_class_labels)
from attack import protect_full_image, compute_psnr, verify_against_models

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL, MODEL_TYPE = load_face_model(DEVICE)
DETECTOR          = load_detector(DEVICE)
DENORM, RENORM    = get_denorm_renorm(MODEL_TYPE)
CLASS_NAMES       = load_class_labels() if MODEL_TYPE == 'resnet50' else None

print(f"\n[Adversarial Shield] Web UI")
print(f"   Model  : {'InceptionResnetV1 (VGGFace2)' if MODEL_TYPE=='facenet' else 'ResNet50 fallback'}")
print(f"   Device : {DEVICE}")
print(f"   Visit  : http://localhost:5000\n")


def _pil_to_b64(pil_img, size=None):
    if size:
        pil_img = pil_img.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def _pil_to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def _safe(v):
    if isinstance(v, torch.Tensor):      return float(v.item())
    if isinstance(v, (np.floating, np.integer)): return float(v)
    return v


@app.route("/")
def index():
    return render_template("index.html", model_type=MODEL_TYPE, device=str(DEVICE))


@app.route("/protect", methods=["POST"])
def protect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    epsilon    = max(0.001, min(0.30, float(request.form.get("epsilon", 0.02))))
    steps      = max(1,     min(50,   int(request.form.get("steps",   10))))
    use_fgsm   = request.form.get("use_fgsm", "0") == "1"
    run_verify = request.form.get("verify",   "0") == "1"

    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    # Detect face + preprocess
    image_tensor, face_pil, face_detected, pil_original, face_box = \
        preprocess_for_facenet(pil_img, DETECTOR, DEVICE)

    target_label = None
    if MODEL_TYPE == 'resnet50':
        with torch.no_grad():
            target_label = MODEL(image_tensor).argmax(dim=1).item()

    # Attack → returns full protected image + perturbed_tensor for verify
    protected_pil, face_attacked_pil, perturbed_tensor, metrics = protect_full_image(
        MODEL, MODEL_TYPE, pil_original,
        DENORM, RENORM, epsilon, steps, DEVICE,
        target_label=target_label, use_fgsm=use_fgsm,
        face_tensor=image_tensor, face_box=face_box,
    )

    # PSNR on full image
    orig_np = np.array(pil_original)
    prot_np = np.array(protected_pil)
    psnr    = compute_psnr(orig_np, prot_np)
    psnr    = 99.99 if psnr == float("inf") else psnr

    if MODEL_TYPE == 'resnet50' and CLASS_NAMES:
        metrics['orig_class_name'] = CLASS_NAMES[metrics['orig_class']]
        metrics['adv_class_name']  = CLASS_NAMES[metrics['adv_class']]

    safe_metrics = {k: _safe(v) for k, v in metrics.items()}

    # Multi-model verification — pass BOTH original and PERTURBED tensors
    verify_results = []
    if run_verify and MODEL_TYPE == 'facenet':
        verify_results = verify_against_models(
            orig_face_tensor=image_tensor,          # original (before attack)
            perturbed_face_tensor=perturbed_tensor, # after attack
            device=DEVICE,
        )
        verify_results = [{k: _safe(v) for k, v in r.items()} for r in verify_results]

    return jsonify({
        "original_b64":      _pil_to_b64(face_pil),
        "face_attacked_b64": _pil_to_b64(face_attacked_pil),
        "protected_b64":     base64.b64encode(_pil_to_bytes(protected_pil)).decode(),
        "face_detected":     bool(face_detected),
        "face_box":          list(face_box) if face_box else None,
        "model_type":        MODEL_TYPE,
        "epsilon":           epsilon,
        "steps":             steps,
        "psnr":              round(float(psnr), 2),
        "metrics":           safe_metrics,
        "verify_results":    verify_results,
    })


@app.route("/download", methods=["POST"])
def download():
    data = request.get_json()
    return send_file(
        io.BytesIO(base64.b64decode(data["image_b64"])),
        mimetype="image/png", as_attachment=True,
        download_name="protected.png",
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
