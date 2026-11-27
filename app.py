import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
from transformers import CLIPVisionModel

st.set_page_config(
    page_title="Multi-Layer CLIP Deepfake Triage",
    page_icon="🛡️",
    layout="wide",
)

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CLIP_MEAN = [0.4815, 0.4578, 0.4082]
CLIP_STD = [0.2686, 0.2613, 0.2758]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_CLIP_HEAD_PATHS = [
    "weights/clip_vit_l14_multilayer_head.pth",
    "clip_vit_l14_multilayer_head.pth",
]
DEFAULT_SAM_CNN_PATHS = [
    "weights/best_efficientnet_b4_sam.pth",
    "best_efficientnet_b4_sam.pth",
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


@st.cache_resource(show_spinner="Loading CLIP ViT-L/14 backbone...")
def load_clip_backbone(device_str: str) -> CLIPVisionModel:
    dev = torch.device(device_str)
    model = CLIPVisionModel.from_pretrained(CLIP_MODEL_ID, attn_implementation="eager")
    model.eval()
    model.to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class EffNetB4SAM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = models.efficientnet_b4(weights=None)
        self.features = base.features
        self.sam = SpatialAttention(kernel_size=7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_feat = base.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_feat, 1),
        )

    def forward_with_cam(self, x: torch.Tensor):
        feat = self.features(x)
        refined = self.sam(feat)
        pooled = self.pool(refined).flatten(1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits, refined


@st.cache_resource(show_spinner="Loading SAM CNN baseline (optional)...")
def load_sam_cnn(path: str | None, device_str: str):
    if not path or not os.path.exists(path):
        return None
    dev = torch.device(device_str)
    model = EffNetB4SAM()
    ckpt = torch.load(path, map_location=dev)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(dev)
    return model


def build_linear_from_sklearn_or_torch(ckpt: object, expected_dim: int = 5120, device: torch.device = torch.device("cpu")) -> nn.Module:
    layer = nn.Linear(expected_dim, 1)

    if isinstance(ckpt, nn.Module):
        ckpt.eval()
        return ckpt.to(device)

    if hasattr(ckpt, "coef_") and hasattr(ckpt, "intercept_"):
        w = torch.tensor(np.asarray(ckpt.coef_, dtype=np.float32)).reshape(1, -1)
        b = torch.tensor(np.asarray(ckpt.intercept_, dtype=np.float32)).reshape(1)
        with torch.no_grad():
            layer.weight.copy_(w)
            layer.bias.copy_(b)
        return layer.eval().to(device)

    if isinstance(ckpt, dict):
        if "coef_" in ckpt and "intercept_" in ckpt:
            w = torch.tensor(np.asarray(ckpt["coef_"], dtype=np.float32)).reshape(1, -1)
            b = torch.tensor(np.asarray(ckpt["intercept_"], dtype=np.float32)).reshape(1)
            with torch.no_grad():
                layer.weight.copy_(w)
                layer.bias.copy_(b)
            return layer.eval().to(device)

        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        if isinstance(state, dict):
            w_key = next((k for k in state if k.endswith("weight")), None)
            b_key = next((k for k in state if k.endswith("bias")), None)
            if w_key is not None and b_key is not None:
                w = state[w_key].float().reshape(1, -1)
                b = state[b_key].float().reshape(1)
                with torch.no_grad():
                    layer.weight.copy_(w)
                    layer.bias.copy_(b)
                return layer.eval().to(device)

    raise ValueError(
        "Unrecognised checkpoint schema in clip_vit_l14_multilayer_head.pth. "
        "Expected nn.Module, state_dict with weight/bias, or LogisticRegression weights."
    )


@st.cache_resource(show_spinner="Loading Multi-Layer CLIP head...")
def load_multilayer_head(path: str, device_str: str) -> nn.Module:
    dev = torch.device(device_str)
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    return build_linear_from_sklearn_or_torch(ckpt, expected_dim=5120, device=dev)


def get_clip_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_cnn_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def extract_multilayer_descriptor(backbone: CLIPVisionModel, img_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = backbone(
        pixel_values=img_tensor,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    h12_cls = l2_norm(hidden_states[12][:, 0, :])
    h16_cls = l2_norm(hidden_states[16][:, 0, :])
    h20_seq = hidden_states[20]
    h20_cls = l2_norm(h20_seq[:, 0, :])
    h20_patch_mean = l2_norm(h20_seq[:, 1:, :].mean(dim=1))
    z24 = l2_norm(outputs.pooler_output)

    fused = torch.cat([h12_cls, h16_cls, h20_cls, h20_patch_mean, z24], dim=-1)
    descriptor = l2_norm(fused, dim=-1)
    return descriptor, outputs.attentions[-1]


@torch.no_grad()
def predict_multilayer_clip(
    image: Image.Image,
    backbone: CLIPVisionModel,
    head: nn.Module,
    device: torch.device,
    use_tta: bool = True,
) -> dict:
    tfm = get_clip_transform()
    rgb_img = image.convert("RGB")
    x_orig = tfm(rgb_img).unsqueeze(0).to(device)

    desc_orig, attn_orig = extract_multilayer_descriptor(backbone, x_orig)
    if use_tta:
        x_flip = tfm(rgb_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
        desc_flip, _ = extract_multilayer_descriptor(backbone, x_flip)
        desc = l2_norm(0.5 * (desc_orig + desc_flip), dim=-1)
    else:
        desc = desc_orig

    logit = head(desc).squeeze()
    prob_real = float(torch.sigmoid(logit).item())
    prob_fake = 1.0 - prob_real

    cls_to_patches = attn_orig[0, :, 0, 1:].mean(dim=0).reshape(1, 1, 16, 16)
    saliency_224 = F.interpolate(cls_to_patches, size=(224, 224), mode="bilinear", align_corners=False)
    sal_np = saliency_224.squeeze().cpu().numpy()
    sal_np = (sal_np - sal_np.min()) / (sal_np.max() - sal_np.min() + 1e-8)

    return {
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "saliency": sal_np,
        "resized_rgb": np.array(rgb_img.resize((224, 224), Image.Resampling.BICUBIC)),
    }


def compute_gradcam_sam(model: EffNetB4SAM, image: Image.Image, device: torch.device) -> tuple[np.ndarray, float]:
    model.zero_grad(set_to_none=True)
    rgb_img = image.convert("RGB")
    x = get_cnn_transform()(rgb_img).unsqueeze(0).to(device)
    x.requires_grad_(True)

    activations = []
    gradients = []

    def fwd_hook(_, __, output):
        activations.append(output)

    def bwd_hook(_, __, grad_output):
        gradients.append(grad_output[0])

    hook_f = model.sam.conv.register_forward_hook(fwd_hook)
    hook_b = model.sam.conv.register_full_backward_hook(bwd_hook)

    try:
        logits, _ = model.forward_with_cam(x)
        prob_real = float(torch.sigmoid(logits).item())
        score = logits.sum()
        score.backward()

        act = activations[0].detach()
        grad = gradients[0].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam_up = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam_np = cam_up.squeeze().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np, prob_real
    finally:
        hook_f.remove()
        hook_b.remove()


def route_three_zone_triage(prob_real: float, auto_flag_below: float = 0.30, auto_pass_above: float = 0.70) -> tuple[str, str]:
    if prob_real < auto_flag_below:
        return (
            "Zone 1: Auto-Flag / Quarantine",
            "High-confidence synthetic indicator (P(Real) < threshold). Route to quarantine / priority compliance review.",
        )
    if prob_real > auto_pass_above:
        return (
            "Zone 3: Auto-Approve / Pass",
            "High-confidence authentic indicator (P(Real) > threshold). Clear for standard distribution.",
        )
    return (
        "Zone 2: Human Moderator Triage Queue",
        "Uncertain / borderline confidence band (0.30–0.70). Escalate to human moderator with saliency inspection.",
    )


def render_overlay(base_rgb: np.ndarray, heatmap_01: np.ndarray, cmap_name: str = "inferno", alpha: float = 0.45) -> Image.Image:
    cmap = plt.get_cmap(cmap_name)
    colored = (cmap(heatmap_01)[:, :, :3] * 255.0).astype(np.float32)
    blended = (1.0 - alpha) * base_rgb.astype(np.float32) + alpha * colored
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def find_existing_path(candidates: list[str]) -> str:
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def main() -> None:
    device = get_device()

    st.title("Multi-Layer CLIP ViT-L/14 Deepfake Detection & 3-Zone Moderation Triage")
    st.caption(
        "MSc Computing Artefact · Teesside University (SCEDT) · Ethics Clearance Ref: 2026 Oct 36071 BATTU"
    )

    with st.sidebar:
        st.header("Model & Triage Settings")
        default_head = find_existing_path(DEFAULT_CLIP_HEAD_PATHS)
        head_path = st.text_input("Multi-Layer CLIP Head Path", value=default_head)

        default_sam = find_existing_path(DEFAULT_SAM_CNN_PATHS)
        sam_path = st.text_input("Optional EffNet-B4 + SAM Path", value=default_sam)

        use_tta = st.toggle("Horizontal-Flip Test-Time Augmentation (TTA)", value=True)
        auto_flag_th = st.slider("Zone 1 Auto-Flag Ceiling: P(Real) <", 0.10, 0.45, 0.30, 0.05)
        auto_pass_th = st.slider("Zone 3 Auto-Pass Floor: P(Real) >", 0.55, 0.90, 0.70, 0.05)
        overlay_alpha = st.slider("Heatmap Overlay Opacity", 0.15, 0.80, 0.45, 0.05)

        st.divider()
        st.markdown("**Reported Benchmarks (Model 4)**")
        st.markdown(
            "- **140k Test:** 99.37% Acc (0.9995 AUC)
"
            "- **FF++ C23 (Video-Safe):** 82.27% Acc (0.9171 AUC)
"
            "- **CIPLab (Unseen):** 85.15% Acc (0.9183 AUC)
"
            "- **Celeb-DF v2:** 73.10% frame / 70.71% video"
        )

    if not os.path.exists(head_path):
        st.error(
            f"Checkpoint not found at `{head_path}`. Place `clip_vit_l14_multilayer_head.pth` in `weights/` "
            "or update the path in the sidebar."
        )
        st.stop()

    backbone = load_clip_backbone(str(device))
    head = load_multilayer_head(head_path, str(device))
    sam_model = load_sam_cnn(sam_path, str(device))

    uploaded = st.file_uploader(
        "Upload a facial image or extracted video frame (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is None:
        st.info("Upload a face image above to run Multi-Layer CLIP ViT-L/14 inference and three-zone moderation routing.")
        return

    pil_img = Image.open(uploaded)
    result = predict_multilayer_clip(pil_img, backbone, head, device, use_tta=use_tta)
    prob_real = result["prob_real"]
    prob_fake = result["prob_fake"]
    zone_label, zone_desc = route_three_zone_triage(prob_real, auto_flag_th, auto_pass_th)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(Real)", f"{prob_real * 100:.2f}%")
    c2.metric("P(Fake)", f"{prob_fake * 100:.2f}%")
    c3.metric("Binary Verdict (τ = 0.50)", "Real" if prob_real >= 0.50 else "Fake")
    c4.metric("TTA Mode", "Enabled (2-view)" if use_tta else "Disabled (1-view)")

    st.markdown(f"### {zone_label}")
    st.write(zone_desc)

    col_orig, col_clip = st.columns(2)
    with col_orig:
        st.subheader("Input Face (224 × 224)")
        st.image(result["resized_rgb"], use_container_width=True)

    with col_clip:
        st.subheader("CLIP ViT-L/14 CLS-to-Patch Attention")
        clip_overlay = render_overlay(result["resized_rgb"], result["saliency"], cmap_name="inferno", alpha=overlay_alpha)
        st.image(clip_overlay, use_container_width=True)

    if sam_model is not None:
        st.divider()
        st.subheader("Comparison Baseline: EfficientNet-B4 + SAM Grad-CAM")
        cam_map, cam_real = compute_gradcam_sam(sam_model, pil_img, device)
        cam_overlay = render_overlay(result["resized_rgb"], cam_map, cmap_name="jet", alpha=overlay_alpha)
        sc1, sc2 = st.columns(2)
        sc1.metric("EffNet-B4 + SAM P(Real)", f"{cam_real * 100:.2f}%")
        sc2.image(cam_overlay, caption="SAM Convolutional Grad-CAM Overlay", use_container_width=True)


if __name__ == "__main__":
    main()
