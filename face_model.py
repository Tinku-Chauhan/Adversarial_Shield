"""
face_model.py — FaceNet + MTCNN loader.

Returns the face bounding box alongside the image tensor so
the attack can be applied only to the face region.
"""

import torch
import torchvision.transforms as T
from PIL import Image

try:
    from facenet_pytorch import InceptionResnetV1, MTCNN
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False

import torchvision.models as models

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
FACENET_SIZE  = 160


def load_face_model(device):
    if FACENET_AVAILABLE:
        model = InceptionResnetV1(pretrained='vggface2').to(device)
        model.eval()
        print("[model] Loaded InceptionResnetV1 (VGGFace2)")
        return model, 'facenet'
    else:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model   = models.resnet50(weights=weights).to(device)
        model.eval()
        print("[model] Falling back to ResNet50")
        return model, 'resnet50'


def load_detector(device):
    if not FACENET_AVAILABLE:
        return None
    return MTCNN(
        image_size=FACENET_SIZE, margin=20,
        keep_all=False, select_largest=True,
        device=device, post_process=False,
    )


def preprocess_for_facenet(pil_img, detector, device):
    """
    Returns:
        face_tensor  : (1,3,160,160) in [-1,1]
        face_pil     : 160×160 PIL crop for display
        detected     : bool
        pil_original : original PIL at full resolution (unchanged)
        face_box     : (x1,y1,x2,y2) bounding box on the original image, or None
    """
    pil_original = pil_img

    if detector is None:
        t, fp, det = _preprocess_resnet(pil_img, device)
        return t, fp, det, pil_original, None

    # MTCNN with return_prob so we can also get the box
    # Use detect() to get boxes, then extract crop manually for better control
    boxes, probs = detector.detect(pil_img)

    if boxes is None or len(boxes) == 0:
        print("[detector] No face detected — using full image")
        t, fp, _ = _preprocess_facenet_full_image(pil_img, device)
        return t, fp, False, pil_original, None

    # Pick highest confidence box
    best_idx  = int(probs.argmax())
    box       = boxes[best_idx]          # [x1, y1, x2, y2]
    x1, y1, x2, y2 = box

    # Add margin (same as MTCNN default=20)
    margin = 20
    orig_w, orig_h = pil_img.size
    x1m = max(0, int(x1) - margin)
    y1m = max(0, int(y1) - margin)
    x2m = min(orig_w, int(x2) + margin)
    y2m = min(orig_h, int(y2) + margin)
    face_box = (x1m, y1m, x2m, y2m)

    # Crop and resize to 160×160
    face_crop   = pil_img.crop(face_box).resize((FACENET_SIZE, FACENET_SIZE), Image.LANCZOS)
    face_tensor = T.ToTensor()(face_crop).unsqueeze(0).to(device)
    face_tensor = face_tensor * 2.0 - 1.0   # [0,1] → [-1,1]

    display  = ((face_tensor.detach().squeeze(0) + 1.0) / 2.0).clamp(0, 1)
    face_pil = T.ToPILImage()(display.cpu())

    return face_tensor, face_pil, True, pil_original, face_box


def _preprocess_facenet_full_image(pil_img, device):
    tfm = T.Compose([T.Resize((FACENET_SIZE, FACENET_SIZE)), T.ToTensor()])
    t   = (tfm(pil_img).unsqueeze(0).to(device)) * 2.0 - 1.0
    display  = ((t.detach().squeeze(0) + 1.0) / 2.0).clamp(0, 1)
    face_pil = T.ToPILImage()(display.cpu())
    return t, face_pil, False


def _preprocess_resnet(pil_img, device):
    tfm = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    t = tfm(pil_img).unsqueeze(0).to(device)
    return t, pil_img, False


def facenet_denorm(t):
    return ((t + 1.0) / 2.0).clamp(0, 1)

def facenet_renorm(t):
    return t * 2.0 - 1.0

def resnet_denorm(t):
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD,  dtype=t.dtype, device=t.device).view(1,3,1,1)
    return (t * std + mean).clamp(0, 1)

def resnet_renorm(t):
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD,  dtype=t.dtype, device=t.device).view(1,3,1,1)
    return (t - mean) / std

def get_denorm_renorm(model_type):
    if model_type == 'facenet':
        return facenet_denorm, facenet_renorm
    return resnet_denorm, resnet_renorm

def load_class_labels():
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    return weights.meta["categories"]
