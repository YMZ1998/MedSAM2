import os
import importlib
import tempfile
from functools import lru_cache
from glob import glob
from html import escape
from os.path import basename, splitext

import numpy as np
from PIL import Image

from nii_inference import (
    NiftiSegmenter,
    load_nifti_volume,
    overlay_mask_on_slice,
    save_mask_nifti,
    validate_nifti_path,
)


SEGMENTER_CACHE = {}
CANVAS_REALTIME_UPDATE = True


def strip_nii_gz(filename):
    name = basename(filename)
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return splitext(name)[0]


def discover_files(folder, pattern):
    files = sorted(glob(os.path.join(folder, pattern)))
    return [path for path in files if os.path.isfile(path)]


def display_name(path):
    name = basename(path)
    if name.endswith(".yaml"):
        return splitext(name)[0]
    if name.endswith(".pt"):
        return splitext(name)[0]
    return name


def build_file_map(paths):
    return {display_name(path): path for path in paths}


def build_config_map(paths):
    return {
        display_name(path): os.path.join("configs", basename(path)).replace("\\", "/")
        for path in paths
    }


CONFIG_MAP = build_config_map(discover_files(os.path.join("sam2", "configs"), "*.yaml"))
CHECKPOINT_MAP = build_file_map(discover_files("checkpoints", "*.pt"))
DEFAULT_CONFIG_NAME = "sam2.1_hiera_t512"
DEFAULT_CHECKPOINT_NAME = "MedSAM2_latest"


def select_default_name(choices, preferred):
    if preferred in choices:
        return preferred
    if not choices:
        raise ValueError("No choices are available.")
    return choices[0]


@lru_cache(maxsize=1)
def get_torch_status():
    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        return {
            "available": True,
            "device": "cuda" if cuda_available else "cpu",
            "error": "",
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": cuda_available,
        }
    except Exception as exc:
        return {
            "available": False,
            "device": "cpu",
            "error": str(exc),
            "version": "",
            "cuda_available": False,
        }


def materialize_uploaded_nifti(uploaded_file, output_dir):
    if uploaded_file is None:
        raise ValueError("Upload a .nii.gz file first.")
    validate_nifti_path(uploaded_file.name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, basename(uploaded_file.name))
    payload = uploaded_file.getbuffer()
    with open(output_path, "wb") as handle:
        handle.write(payload)
    return output_path


def get_segmenter(config_name, checkpoint_name, device):
    if config_name not in CONFIG_MAP:
        raise ValueError(f"Unknown config selection: {config_name}")
    if checkpoint_name not in CHECKPOINT_MAP:
        raise ValueError(f"Unknown checkpoint selection: {checkpoint_name}")

    key = (CONFIG_MAP[config_name], CHECKPOINT_MAP[checkpoint_name], device)
    if key not in SEGMENTER_CACHE:
        SEGMENTER_CACHE[key] = NiftiSegmenter(
            config_path=key[0],
            checkpoint_path=key[1],
            device=device,
        )
    return SEGMENTER_CACHE[key]


def format_measurement(value):
    return f"{value:.4g}"


def case_summary_items(volume, slice_idx, torch_status, config_name):
    depth, height, width = volume.array.shape
    spacing = volume.image.GetSpacing()
    device = torch_status["device"].upper()
    device_detail = "Ready" if torch_status["available"] else "Unavailable"
    return [
        ("Case", basename(volume.path), "Loaded"),
        ("Dimensions", f"{depth} x {height} x {width}", "(D x H x W)"),
        (
            "Spacing",
            " x ".join(format_measurement(value) for value in spacing),
            "mm",
        ),
        ("Voxel Count", f"{int(np.prod(volume.array.shape)):,}", "voxels"),
        ("Slice", f"{int(slice_idx)} / {depth - 1}", "Current axial index"),
        ("Device", device, device_detail),
        ("Model", config_name, "Checkpoint selected"),
    ]


def thumbnail_indices(slice_idx, depth, radius=3):
    slice_idx = int(slice_idx)
    depth = int(depth)
    if depth <= 0:
        return []
    window = min(depth, radius * 2 + 1)
    start = max(0, min(slice_idx - radius, depth - window))
    return list(range(start, start + window))


def mask_summary(state):
    mask = state["mask"]
    voxels = int(np.count_nonzero(mask))
    spacing = state["volume"].image.GetSpacing()
    volume_cm3 = round(voxels * float(np.prod(spacing)) / 1000.0, 3)
    coverage = round((voxels / float(mask.size)) * 100.0, 2) if mask.size else 0.0
    return {
        "voxels": voxels,
        "volume_cm3": volume_cm3,
        "coverage": coverage,
    }


def product_css():
    return """
    <style>
    :root {
        --med-bg: #f7faff;
        --med-panel: #ffffff;
        --med-panel-soft: #f1f6fd;
        --med-ink: #121f35;
        --med-muted: #66738a;
        --med-border: #dfe7f1;
        --med-blue: #2f6eea;
        --med-blue-soft: #eaf2ff;
        --med-cyan: #11b9bd;
        --med-green: #159447;
        --med-orange: #f59b32;
        --med-shadow: 0 16px 40px rgba(31, 45, 70, 0.08);
    }
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu {
        display: none;
    }
    .stApp {
        background: var(--med-bg);
        color: var(--med-ink);
    }
    .block-container {
        padding: 18px 24px 20px;
        max-width: none;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fbff 0%, #f2f6fc 100%);
        border-right: 1px solid var(--med-border);
        box-shadow: 18px 0 36px rgba(31, 45, 70, 0.045);
        min-width: 310px;
        width: 310px;
    }
    [data-testid="stSidebar"] section {
        padding-top: 18px;
    }
    [data-testid="stSidebarCollapseButton"],
    button[title="Close sidebar"],
    button[data-testid="baseButton-header"],
    button[data-testid="baseButton-headerNoPadding"] {
        display: none;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--med-ink);
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--med-ink);
    }
    .med-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 4px 0 26px;
    }
    .med-logo {
        width: 50px;
        height: 50px;
        border-radius: 13px;
        background: linear-gradient(135deg, #2f75ee 0%, #245bc8 100%);
        color: white;
        display: grid;
        place-items: center;
        font-size: 31px;
        line-height: 1;
        font-weight: 900;
        box-shadow: 0 12px 26px rgba(47, 110, 234, 0.26);
    }
    .med-brand-name {
        font-size: 1.55rem;
        line-height: 1.02;
        font-weight: 830;
        color: var(--med-ink);
    }
    .med-brand-sub {
        display: block;
        color: var(--med-blue);
        font-weight: 800;
    }
    .med-section-label {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #405984;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 20px 0 12px;
    }
    .med-section-label:after {
        content: "";
        flex: 1;
        height: 1px;
        background: var(--med-border);
    }
    .med-upload-card {
        border: 1px solid var(--med-border);
        background: var(--med-panel);
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 10px 24px rgba(31, 45, 70, 0.04);
        margin-bottom: 12px;
    }
    .med-file-card {
        display: grid;
        grid-template-columns: 32px 1fr 22px;
        align-items: center;
        gap: 10px;
        border: 1px solid var(--med-border);
        background: var(--med-panel);
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
    }
    .med-file-icon {
        color: #7384a4;
        font-size: 1.35rem;
    }
    .med-file-name {
        font-size: 0.92rem;
        font-weight: 760;
        color: var(--med-ink);
        overflow-wrap: anywhere;
    }
    .med-file-size {
        font-size: 0.8rem;
        color: var(--med-muted);
        margin-top: 2px;
    }
    .med-check {
        color: var(--med-green);
        font-weight: 900;
        font-size: 1.1rem;
    }
    .med-runtime {
        border: 1px solid #bfe3d0;
        background: #e5f5ea;
        border-radius: 8px;
        padding: 11px 12px;
        margin: 12px 0 18px;
        color: #126337;
        font-size: 0.86rem;
        font-weight: 650;
    }
    .med-runtime.offline {
        border-color: rgba(220, 53, 69, 0.24);
        background: #fff0f1;
        color: #a32635;
    }
    .med-status-grid {
        display: grid;
        grid-template-columns: 1fr 1.1fr 1.1fr 1.1fr 1.35fr 0.8fr 1.2fr;
        gap: 0;
        background: var(--med-panel);
        border-bottom: 1px solid var(--med-border);
        margin: -2px 0 18px;
        box-shadow: 0 1px 0 rgba(31, 45, 70, 0.03);
    }
    .med-status-item {
        padding: 15px 20px 14px;
        min-height: 88px;
        border-right: 1px solid var(--med-border);
    }
    .med-status-label {
        color: #52637c;
        font-size: 0.72rem;
        font-weight: 760;
        margin-bottom: 8px;
    }
    .med-status-value {
        color: var(--med-ink);
        font-size: 0.98rem;
        line-height: 1.2;
        font-weight: 780;
        overflow-wrap: anywhere;
    }
    .med-status-detail {
        color: var(--med-muted);
        font-size: 0.78rem;
        margin-top: 5px;
    }
    .med-viewer-title {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }
    .med-viewer-title h2 {
        margin: 0;
        font-size: 1rem;
        font-weight: 760;
        color: #31518f;
    }
    .med-step {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        background: #e8f1ff;
        color: var(--med-blue);
        font-weight: 850;
        font-size: 1rem;
    }
    .med-viewer-caption {
        color: var(--med-muted);
        font-size: 0.82rem;
        margin-bottom: 10px;
    }
    .med-panel-note {
        border-left: 3px solid var(--med-blue);
        background: #f0f6ff;
        color: #1b4f99;
        padding: 10px 12px;
        border-radius: 7px;
        margin: 12px 0;
        font-size: 0.88rem;
    }
    .med-panel-note.warn {
        border-left-color: var(--med-orange);
        background: #fff6ed;
        color: #8a431e;
    }
    .med-viewer-placeholder {
        min-height: 520px;
        border-radius: 8px;
        background:
            radial-gradient(circle at 50% 48%, rgba(91, 103, 116, 0.34), transparent 34%),
            linear-gradient(180deg, #070d15 0%, #02060a 100%);
        border: 1px solid #111827;
        box-shadow: 0 18px 48px rgba(14, 27, 39, 0.16);
        display: grid;
        place-items: center;
        color: #8fa0b4;
        font-weight: 760;
    }
    .med-empty {
        background: var(--med-panel);
        border: 1px dashed #bed0e4;
        border-radius: 8px;
        padding: 38px;
        margin-top: 20px;
        box-shadow: var(--med-shadow);
    }
    .med-empty h2 {
        margin: 0 0 8px;
        font-size: 1.35rem;
    }
    .med-empty p {
        color: var(--med-muted);
        margin: 0;
        max-width: 720px;
    }
    .med-results {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0;
        border: 1px solid var(--med-border);
        border-radius: 8px;
        overflow: hidden;
        background: var(--med-panel);
        margin-top: 10px;
    }
    .med-result-item {
        background: var(--med-panel);
        border-right: 1px solid var(--med-border);
        padding: 14px 16px;
    }
    .med-result-label {
        color: var(--med-muted);
        font-size: 0.72rem;
        font-weight: 760;
    }
    .med-result-value {
        color: var(--med-ink);
        font-size: 1.05rem;
        font-weight: 780;
        margin-top: 5px;
    }
    .med-thumb-strip {
        display: grid;
        grid-template-columns: 34px repeat(7, minmax(50px, 1fr)) 34px;
        gap: 8px;
        align-items: stretch;
        background: #050b13;
        border: 1px solid #101a27;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 18px 48px rgba(14, 27, 39, 0.14);
        margin-top: 10px;
    }
    .med-thumb-nav {
        display: grid;
        place-items: center;
        color: #eaf1fb;
        font-size: 1.45rem;
        font-weight: 700;
    }
    .med-thumb {
        border: 1px solid #132235;
        border-radius: 7px;
        color: #eaf1fb;
        background:
            radial-gradient(circle at 50% 35%, rgba(150, 160, 170, 0.45), rgba(50, 58, 66, 0.35) 34%, rgba(13, 20, 28, 0.94) 70%),
            #07101b;
        min-height: 72px;
        display: flex;
        align-items: flex-end;
        justify-content: center;
        padding: 8px;
        font-weight: 760;
    }
    .med-thumb.active {
        border: 2px solid var(--med-blue);
        box-shadow: inset 0 0 0 1px rgba(47, 110, 234, 0.4);
    }
    .med-info-strip {
        margin-top: 18px;
        background: #eaf3ff;
        color: #235ec6;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.92rem;
        border: 1px solid #d6e7ff;
    }
    .med-controls-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 10px;
    }
    .med-quick-card {
        border: 1px solid var(--med-border);
        border-radius: 8px;
        padding: 14px 16px;
        background: var(--med-panel);
        min-height: 90px;
    }
    .med-quick-title {
        color: #52637c;
        font-size: 0.82rem;
        font-weight: 780;
        margin-bottom: 12px;
    }
    .med-swatch {
        width: 34px;
        height: 34px;
        border-radius: 7px;
        display: inline-block;
        background: var(--med-cyan);
        box-shadow: inset 0 0 0 2px rgba(255, 255, 255, 0.85);
        border: 1px solid #71d8d8;
        vertical-align: middle;
        margin-right: 10px;
    }
    .med-mini-button {
        display: inline-grid;
        place-items: center;
        width: 34px;
        height: 34px;
        border: 1px solid var(--med-border);
        border-radius: 7px;
        margin-right: 6px;
        color: #31445f;
        background: #fbfdff;
    }
    div[data-testid="stImage"] img,
    iframe[title="streamlit_drawable_canvas.st_canvas"] {
        border-radius: 8px;
        border: 1px solid #111827;
        box-shadow: 0 18px 48px rgba(14, 27, 39, 0.16);
        background: #02060a;
    }
    div[data-testid="stImage"] {
        margin-bottom: 0;
    }
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 7px;
        font-weight: 750;
        min-height: 2.55rem;
        width: 100%;
    }
    .stButton > button[kind="primary"] {
        background: var(--med-blue);
        border-color: var(--med-blue);
    }
    [data-testid="stSlider"] {
        margin-top: -10px;
        margin-bottom: 12px;
    }
    @media (max-width: 1100px) {
        .med-status-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 720px) {
        .block-container {
            padding: 14px;
        }
        .med-status-grid,
        .med-results,
        .med-controls-row {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """


def render_metric_grid(items):
    cards = []
    for label, value, detail in items:
        cards.append(
            '<div class="med-status-item">'
            f'<div class="med-status-label">{escape(label)}</div>'
            f'<div class="med-status-value">{escape(value)}</div>'
            f'<div class="med-status-detail">{escape(detail)}</div>'
            "</div>"
        )
    return f'<div class="med-status-grid">{"".join(cards)}</div>'


def render_mask_summary(summary):
    return f"""
    <div class="med-results">
        <div class="med-result-item">
            <div class="med-result-label">Mask voxels</div>
            <div class="med-result-value">{summary["voxels"]:,}</div>
        </div>
        <div class="med-result-item">
            <div class="med-result-label">Volume</div>
            <div class="med-result-value">{summary["volume_cm3"]:.3f} cm3</div>
        </div>
        <div class="med-result-item">
            <div class="med-result-label">Coverage</div>
            <div class="med-result-value">{summary["coverage"]:.2f}%</div>
        </div>
    </div>
    """


def render_sidebar_brand():
    return """
    <div class="med-brand">
        <div class="med-logo">✚</div>
        <div class="med-brand-name">MedSAM2<br><span class="med-brand-sub">NIfTI Studio</span></div>
    </div>
    """


def render_loaded_file_card(uploaded_name, uploaded_size):
    if not uploaded_name:
        return ""
    size_mb = uploaded_size / (1024 * 1024) if uploaded_size else 0
    return f"""
    <div class="med-file-card">
        <div class="med-file-icon">□</div>
        <div>
            <div class="med-file-name">{escape(uploaded_name)}</div>
            <div class="med-file-size">{size_mb:.1f}MB</div>
        </div>
        <div class="med-check">✓</div>
    </div>
    """


def render_viewer_title(step, title):
    return f"""
    <div class="med-viewer-title">
        <div class="med-step">{step}</div>
        <h2>{escape(title)}</h2>
    </div>
    """


def render_thumbnail_strip(indices, active_idx):
    thumbs = ['<div class="med-thumb-nav">‹</div>']
    for idx in indices:
        active = " active" if int(idx) == int(active_idx) else ""
        thumbs.append(f'<div class="med-thumb{active}">{int(idx)}</div>')
    thumbs.append('<div class="med-thumb-nav">›</div>')
    return f'<div class="med-thumb-strip">{"".join(thumbs)}</div>'


def render_quick_controls():
    return """
    <div class="med-quick-card">
        <div class="med-quick-title">Quick Controls</div>
        <span class="med-swatch"></span>
        <span class="med-mini-button">◎</span>
        <span class="med-mini-button">◉</span>
        <span style="color:#52637c;font-weight:760;">50%</span>
    </div>
    """


def render_placeholder_status():
    return render_metric_grid(
        [
            ("Case", "No study", "Waiting"),
            ("Dimensions", "- x - x -", "(H x W x D)"),
            ("Spacing", "- x - x -", "mm"),
            ("Voxel Count", "-", "voxels"),
            ("Slice", "- / -", "Load volume"),
            ("Device", "CUDA", "Ready"),
            ("Model", DEFAULT_CONFIG_NAME, "Ready"),
        ]
    )


def create_session_state(volume):
    return {
        "volume": volume,
        "mask": np.zeros(volume.array.shape, dtype=np.uint8),
        "output_dir": tempfile.mkdtemp(prefix="medsam2_nii_"),
        "output_path": None,
        "last_bbox": None,
    }


def image_for_slice(state, slice_idx, with_mask=True):
    volume = state["volume"]
    slice_idx = int(slice_idx)
    image_slice = volume.preview[slice_idx]
    if with_mask:
        return overlay_mask_on_slice(image_slice, state["mask"][slice_idx])
    return np.repeat(image_slice[..., None], 3, axis=-1)


def compute_display_size(image_shape, max_width=768):
    height, width = image_shape[:2]
    if width <= max_width:
        return width, height
    scale = max_width / float(width)
    return max(1, int(width * scale)), max(1, int(height * scale))


def bbox_from_canvas(canvas_json, display_size, source_shape):
    objects = (canvas_json or {}).get("objects") or []
    if not objects:
        raise ValueError("Draw a rectangle over the target before segmenting.")

    item = objects[-1]
    if item.get("type") != "rect":
        raise ValueError("Use the rectangle drawing mode to mark the target.")

    display_width, display_height = display_size
    source_height, source_width = source_shape[:2]
    scale_x = source_width / float(display_width)
    scale_y = source_height / float(display_height)

    left = float(item.get("left", 0.0))
    top = float(item.get("top", 0.0))
    width = float(item.get("width", 0.0)) * float(item.get("scaleX", 1.0))
    height = float(item.get("height", 0.0)) * float(item.get("scaleY", 1.0))

    x0 = int(max(0, min(source_width - 1, round(left * scale_x))))
    y0 = int(max(0, min(source_height - 1, round(top * scale_y))))
    x1 = int(max(0, min(source_width - 1, round((left + width) * scale_x))))
    y1 = int(max(0, min(source_height - 1, round((top + height) * scale_y))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("The rectangle prompt is too small.")
    return np.array([x0, y0, x1, y1], dtype=np.int64)


def segment_current_slice(state, bbox, slice_idx, config_name, checkpoint_name, device):
    torch_status = get_torch_status()
    if not torch_status["available"]:
        raise RuntimeError(
            "PyTorch is not available in the Python environment running Streamlit. "
            f"Start this app from a MedSAM2 environment with working torch. Details: {torch_status['error']}"
        )
    segmenter = get_segmenter(config_name, checkpoint_name, device)
    volume = state["volume"]
    mask = segmenter.segment_with_box(volume, int(slice_idx), bbox)
    output_name = f"{strip_nii_gz(volume.path)}_mask.nii.gz"
    output_path = os.path.join(state["output_dir"], output_name)
    save_mask_nifti(mask, volume, output_path)
    state["mask"] = mask
    state["output_path"] = output_path
    state["last_bbox"] = [int(value) for value in bbox]
    return output_path


def reset_mask(state):
    state["mask"] = np.zeros(state["volume"].array.shape, dtype=np.uint8)
    state["output_path"] = None
    state["last_bbox"] = None


def get_streamlit_canvas():
    try:
        from streamlit_drawable_canvas import st_canvas

        return st_canvas
    except Exception:
        return None


def main():
    import streamlit as st

    st.set_page_config(
        page_title="MedSAM2 NIfTI Studio",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(product_css(), unsafe_allow_html=True)

    if not CONFIG_MAP:
        st.error("No config files found under sam2/configs.")
        st.stop()
    if not CHECKPOINT_MAP:
        st.error("No checkpoint files found under checkpoints.")
        st.stop()

    if "nii_state" not in st.session_state:
        st.session_state.nii_state = None
    if "upload_workspace" not in st.session_state:
        st.session_state.upload_workspace = tempfile.mkdtemp(prefix="medsam2_upload_")
    if "current_bbox" not in st.session_state:
        st.session_state.current_bbox = None
    if "segment_requested" not in st.session_state:
        st.session_state.segment_requested = False

    torch_status = get_torch_status()
    with st.sidebar:
        st.markdown(render_sidebar_brand(), unsafe_allow_html=True)
        st.markdown('<div class="med-section-label">Upload & Data</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("NIfTI file (.nii, .nii.gz)", label_visibility="visible")

        uploaded_name = st.session_state.get("loaded_upload_name")
        uploaded_size = st.session_state.get("loaded_upload_size", 0)
        st.markdown(
            render_loaded_file_card(uploaded_name, uploaded_size),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="med-section-label">Model & Checkpoint</div>', unsafe_allow_html=True)
        config_choices = list(CONFIG_MAP.keys())
        checkpoint_choices = list(CHECKPOINT_MAP.keys())
        default_config = select_default_name(config_choices, DEFAULT_CONFIG_NAME)
        default_checkpoint = select_default_name(checkpoint_choices, DEFAULT_CHECKPOINT_NAME)
        config_name = st.selectbox(
            "Model",
            config_choices,
            index=config_choices.index(default_config),
        )
        checkpoint_name = st.selectbox(
            "Checkpoint",
            checkpoint_choices,
            index=checkpoint_choices.index(default_checkpoint),
        )

        st.markdown('<div class="med-section-label">Device</div>', unsafe_allow_html=True)
        device = st.selectbox(
            "Device",
            ["cuda", "cpu"],
            index=0 if torch_status["device"] == "cuda" else 1,
            disabled=not torch_status["available"],
        )
        runtime_class = "med-runtime" if torch_status["available"] else "med-runtime offline"
        runtime_text = (
            f'● PyTorch {torch_status["version"]}<br>default device: {torch_status["device"]}'
            if torch_status["available"]
            else "PyTorch unavailable | segmentation disabled"
        )
        st.markdown(f'<div class="{runtime_class}">{runtime_text}</div>', unsafe_allow_html=True)
        if not torch_status["available"]:
            with st.expander("PyTorch error"):
                st.code(torch_status["error"])

        if uploaded_file is not None:
            current_name = st.session_state.get("loaded_upload_name")
            if current_name != uploaded_file.name:
                try:
                    input_path = materialize_uploaded_nifti(
                        uploaded_file,
                        st.session_state.upload_workspace,
                    )
                    st.session_state.nii_state = create_session_state(load_nifti_volume(input_path))
                    st.session_state.loaded_upload_name = uploaded_file.name
                    st.session_state.loaded_upload_size = getattr(uploaded_file, "size", 0)
                    st.session_state.slice_idx = st.session_state.nii_state["volume"].array.shape[0] // 2
                    st.session_state.current_bbox = None
                    st.success("Volume loaded.")
                except Exception as exc:
                    st.session_state.nii_state = None
                    st.error(str(exc))

        st.markdown('<div class="med-section-label">Actions</div>', unsafe_allow_html=True)
        segment_disabled = st.session_state.nii_state is None or not torch_status["available"]
        if st.button("Segment", type="primary", disabled=segment_disabled):
            st.session_state.segment_requested = True
        if st.session_state.nii_state is not None and st.button("Reset Mask"):
            reset_mask(st.session_state.nii_state)
            st.session_state.current_bbox = None
            st.info("Mask reset.")
        if (
            st.session_state.nii_state is not None
            and st.session_state.nii_state.get("output_path")
            and os.path.exists(st.session_state.nii_state["output_path"])
        ):
            with open(st.session_state.nii_state["output_path"], "rb") as handle:
                st.download_button(
                    "Download Mask",
                    data=handle,
                    file_name=basename(st.session_state.nii_state["output_path"]),
                    mime="application/gzip",
                )
        else:
            st.download_button(
                "Download Mask",
                data=b"",
                file_name="mask.nii.gz",
                disabled=True,
            )
        with st.expander("Advanced Options"):
            st.caption("Canvas prompt mode: rectangle")
            st.caption("Propagation: full volume")

    state = st.session_state.nii_state
    if state is None:
        st.markdown(render_placeholder_status(), unsafe_allow_html=True)
        empty_left, empty_right = st.columns(2)
        with empty_left:
            st.markdown(render_viewer_title(1, "Prompt (Draw Rectangle)"), unsafe_allow_html=True)
            st.markdown(
                '<div class="med-viewer-placeholder">Load a NIfTI volume to begin</div>',
                unsafe_allow_html=True,
            )
            st.markdown(render_thumbnail_strip([], 0), unsafe_allow_html=True)
        with empty_right:
            st.markdown(render_viewer_title(2, "Segmentation Overlay"), unsafe_allow_html=True)
            st.markdown(
                '<div class="med-viewer-placeholder">Mask overlay will appear here</div>',
                unsafe_allow_html=True,
            )
            stats_col, controls_col = st.columns(2)
            with stats_col:
                st.markdown(
                    render_mask_summary({"voxels": 0, "volume_cm3": 0, "coverage": 0}),
                    unsafe_allow_html=True,
                )
            with controls_col:
                st.markdown(render_quick_controls(), unsafe_allow_html=True)
        st.markdown(
            '<div class="med-info-strip">ⓘ Import a .nii.gz study from the left panel, then draw a rectangle on the prompt viewer and click Segment.</div>',
            unsafe_allow_html=True,
        )
        return

    volume = state["volume"]

    max_slice = int(volume.array.shape[0] - 1)
    current_slice = int(st.session_state.get("slice_idx", max_slice // 2))
    st.markdown(
        render_metric_grid(
            case_summary_items(volume, current_slice, torch_status, config_name)
        ),
        unsafe_allow_html=True,
    )
    slice_idx = st.slider(
        "Slice index",
        min_value=0,
        max_value=max_slice,
        value=current_slice,
        step=1,
        key="slice_idx",
        label_visibility="collapsed",
    )
    mask_stats = mask_summary(state)

    prompt_rgb = image_for_slice(state, slice_idx, with_mask=False)
    overlay_rgb = image_for_slice(state, slice_idx, with_mask=True)
    display_size = compute_display_size(prompt_rgb.shape)
    display_width, display_height = display_size

    prompt_col, overlay_col = st.columns(2)
    with prompt_col:
        st.markdown(
            render_viewer_title(1, "Prompt (Draw Rectangle)"),
            unsafe_allow_html=True,
        )
        st_canvas = get_streamlit_canvas()
        bbox = None
        if st_canvas is not None:
            canvas_result = st_canvas(
                fill_color="rgba(255, 80, 40, 0.25)",
                stroke_width=2,
                stroke_color="#ff5028",
                background_image=Image.fromarray(prompt_rgb),
                drawing_mode="rect",
                width=display_width,
                height=display_height,
                update_streamlit=CANVAS_REALTIME_UPDATE,
                key=f"canvas_{slice_idx}",
            )
            if canvas_result.json_data:
                try:
                    bbox = bbox_from_canvas(
                        canvas_result.json_data,
                        display_size=display_size,
                        source_shape=prompt_rgb.shape,
                    )
                    st.session_state.current_bbox = bbox
                    st.markdown(
                        f'<div class="med-panel-note">Current box: {bbox.tolist()}</div>',
                        unsafe_allow_html=True,
                    )
                except ValueError as exc:
                    st.markdown(
                        f'<div class="med-panel-note warn">{str(exc)}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.image(prompt_rgb, caption="Install streamlit-drawable-canvas for drawing.")
            st.warning("streamlit-drawable-canvas is not installed. Enter a box manually.")
            height, width = prompt_rgb.shape[:2]
            x0 = st.number_input("x_min", min_value=0, max_value=width - 1, value=0)
            y0 = st.number_input("y_min", min_value=0, max_value=height - 1, value=0)
            x1 = st.number_input("x_max", min_value=0, max_value=width - 1, value=width - 1)
            y1 = st.number_input("y_max", min_value=0, max_value=height - 1, value=height - 1)
            if x1 > x0 and y1 > y0:
                bbox = np.array([x0, y0, x1, y1], dtype=np.int64)
                st.session_state.current_bbox = bbox

        if st.session_state.segment_requested:
            active_bbox = bbox if bbox is not None else st.session_state.current_bbox
            st.session_state.segment_requested = False
            if active_bbox is None:
                st.error("Draw a rectangle or enter a valid box first.")
            else:
                with st.spinner("Running MedSAM2 propagation..."):
                    try:
                        output_path = segment_current_slice(
                            state,
                            active_bbox,
                            slice_idx,
                            config_name,
                            checkpoint_name,
                            device,
                        )
                        st.success(f"Segmentation complete. Box: {state['last_bbox']}")
                        with open(output_path, "rb") as handle:
                            st.download_button(
                                "Download mask (.nii.gz)",
                                data=handle,
                                file_name=basename(output_path),
                                mime="application/gzip",
                            )
                    except Exception as exc:
                        st.error(str(exc))
        st.markdown(
            render_thumbnail_strip(thumbnail_indices(slice_idx, volume.array.shape[0]), slice_idx),
            unsafe_allow_html=True,
        )

    with overlay_col:
        st.markdown(
            render_viewer_title(2, "Segmentation Overlay"),
            unsafe_allow_html=True,
        )
        st.image(overlay_rgb, caption="Current slice overlay")
        stats_col, controls_col = st.columns(2)
        with stats_col:
            st.markdown(render_mask_summary(mask_stats), unsafe_allow_html=True)
        with controls_col:
            st.markdown(render_quick_controls(), unsafe_allow_html=True)

    st.markdown(
        '<div class="med-info-strip">ⓘ Draw a rectangle on the left image to define the target region. Click Segment to generate the mask.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
