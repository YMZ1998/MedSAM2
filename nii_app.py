import os
import importlib
import tempfile
from functools import lru_cache
from glob import glob
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


def default_device():
    return get_torch_status()["device"]


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


def volume_info(volume):
    spacing = ", ".join(f"{value:.4g}" for value in volume.image.GetSpacing())
    return (
        f"Loaded {basename(volume.path)} | "
        f"shape (D, H, W)={tuple(volume.array.shape)} | "
        f"spacing ({spacing})"
    )


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
        ("Slice", f"{int(slice_idx)} / {depth - 1}", "Current axial index"),
        ("Device", device, device_detail),
        ("Model", config_name, "Checkpoint selected"),
    ]


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
        --med-bg: #f5f8fb;
        --med-panel: #ffffff;
        --med-panel-soft: #eef5f8;
        --med-ink: #132233;
        --med-muted: #647487;
        --med-border: #d9e3ea;
        --med-blue: #1f6feb;
        --med-cyan: #1f9bb4;
        --med-green: #198754;
        --med-orange: #e86134;
        --med-shadow: 0 18px 46px rgba(24, 39, 58, 0.10);
    }
    .stApp {
        background: var(--med-bg);
        color: var(--med-ink);
    }
    .block-container {
        padding-top: 2.8rem;
        padding-bottom: 2.4rem;
        max-width: 1480px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eef5f8 0%, #e8eef4 100%);
        border-right: 1px solid var(--med-border);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label {
        color: var(--med-ink);
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--med-ink);
    }
    .med-app-header {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 24px;
        margin-bottom: 18px;
        padding-bottom: 18px;
        border-bottom: 1px solid var(--med-border);
    }
    .med-app-header h1 {
        margin: 0;
        font-size: 2.15rem;
        line-height: 1.1;
        font-weight: 760;
    }
    .med-header-note {
        color: var(--med-muted);
        font-size: 0.9rem;
        text-align: right;
        max-width: 360px;
    }
    .med-sidebar-title {
        font-size: 1.05rem;
        font-weight: 760;
        margin: 0.4rem 0 0.9rem;
    }
    .med-section-label {
        color: var(--med-muted);
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 1.2rem 0 0.45rem;
    }
    .med-runtime {
        border: 1px solid rgba(31, 155, 180, 0.28);
        background: #e7f6f2;
        border-radius: 8px;
        padding: 12px 13px;
        margin-bottom: 14px;
        color: #0f5d44;
        font-size: 0.86rem;
        font-weight: 650;
    }
    .med-runtime.offline {
        border-color: rgba(220, 53, 69, 0.24);
        background: #fff0f1;
        color: #a32635;
    }
    .med-summary-grid {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 10px;
        margin: 12px 0 20px;
    }
    .med-summary-item {
        background: var(--med-panel);
        border: 1px solid var(--med-border);
        border-radius: 8px;
        padding: 12px 13px;
        box-shadow: 0 10px 22px rgba(24, 39, 58, 0.045);
        min-height: 86px;
    }
    .med-summary-label {
        color: var(--med-muted);
        font-size: 0.72rem;
        font-weight: 760;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 8px;
    }
    .med-summary-value {
        color: var(--med-ink);
        font-size: 1.03rem;
        line-height: 1.2;
        font-weight: 780;
        overflow-wrap: anywhere;
    }
    .med-summary-detail {
        color: var(--med-muted);
        font-size: 0.78rem;
        margin-top: 5px;
    }
    .med-workflow {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
        margin: 2px 0 16px;
    }
    .med-workflow-step {
        border: 1px solid var(--med-border);
        background: rgba(255, 255, 255, 0.72);
        border-radius: 8px;
        padding: 10px 12px;
        color: var(--med-muted);
        font-size: 0.82rem;
        font-weight: 700;
    }
    .med-workflow-step.active {
        border-color: rgba(31, 111, 235, 0.34);
        background: #eef5ff;
        color: var(--med-blue);
    }
    .med-viewer-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    .med-viewer-title h2 {
        margin: 0;
        font-size: 1.08rem;
        font-weight: 760;
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
    .med-empty {
        background: var(--med-panel);
        border: 1px dashed #b8c7d4;
        border-radius: 8px;
        padding: 34px;
        margin-top: 18px;
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
        gap: 8px;
        margin-top: 12px;
    }
    .med-result-item {
        background: #0f1b27;
        border: 1px solid #263746;
        border-radius: 8px;
        padding: 11px 12px;
    }
    .med-result-label {
        color: #95a7b8;
        font-size: 0.72rem;
        font-weight: 760;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .med-result-value {
        color: #f3f8fb;
        font-size: 1.05rem;
        font-weight: 780;
        margin-top: 5px;
    }
    div[data-testid="stImage"] img,
    iframe[title="streamlit_drawable_canvas.st_canvas"] {
        border-radius: 8px;
        border: 1px solid #111827;
        box-shadow: 0 18px 48px rgba(14, 27, 39, 0.16);
        background: #02060a;
    }
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 7px;
        font-weight: 750;
        min-height: 2.55rem;
    }
    @media (max-width: 1100px) {
        .med-summary-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .med-workflow {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .med-app-header {
            display: block;
        }
        .med-header-note {
            text-align: left;
            margin-top: 8px;
        }
    }
    @media (max-width: 720px) {
        .block-container {
            padding-top: 1.2rem;
        }
        .med-summary-grid,
        .med-results {
            grid-template-columns: 1fr;
        }
        .med-workflow {
            grid-template-columns: 1fr;
        }
        .med-app-header h1 {
            font-size: 1.55rem;
        }
    }
    </style>
    """


def render_metric_grid(items):
    cards = []
    for label, value, detail in items:
        cards.append(
            f"""
            <div class="med-summary-item">
                <div class="med-summary-label">{label}</div>
                <div class="med-summary-value">{value}</div>
                <div class="med-summary-detail">{detail}</div>
            </div>
            """
        )
    return f'<div class="med-summary-grid">{"".join(cards)}</div>'


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
        initial_sidebar_state="collapsed",
    )
    st.markdown(product_css(), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="med-app-header">
            <div>
                <h1>NIfTI Interactive Segmentation</h1>
            </div>
            <div class="med-header-note">
                Clinical imaging workspace for box-prompted 3D propagation and mask export.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    torch_status = get_torch_status()
    with st.sidebar:
        st.markdown('<div class="med-sidebar-title">Control Console</div>', unsafe_allow_html=True)
        runtime_class = "med-runtime" if torch_status["available"] else "med-runtime offline"
        runtime_text = (
            f'PyTorch {torch_status["version"]} | default device: {torch_status["device"]}'
            if torch_status["available"]
            else "PyTorch unavailable | segmentation disabled"
        )
        st.markdown(f'<div class="{runtime_class}">{runtime_text}</div>', unsafe_allow_html=True)
        if not torch_status["available"]:
            with st.expander("PyTorch error"):
                st.code(torch_status["error"])

        st.markdown('<div class="med-section-label">Data</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Input NIfTI (.nii.gz)", label_visibility="visible")

        st.markdown('<div class="med-section-label">Model</div>', unsafe_allow_html=True)
        config_choices = list(CONFIG_MAP.keys())
        checkpoint_choices = list(CHECKPOINT_MAP.keys())
        default_config = select_default_name(config_choices, DEFAULT_CONFIG_NAME)
        default_checkpoint = select_default_name(checkpoint_choices, DEFAULT_CHECKPOINT_NAME)
        config_name = st.selectbox(
            "Config",
            config_choices,
            index=config_choices.index(default_config),
        )
        checkpoint_name = st.selectbox(
            "Checkpoint",
            checkpoint_choices,
            index=checkpoint_choices.index(default_checkpoint),
        )
        device = st.selectbox(
            "Device",
            ["cuda", "cpu"],
            index=0 if torch_status["device"] == "cuda" else 1,
            disabled=not torch_status["available"],
        )

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
                    st.session_state.slice_idx = st.session_state.nii_state["volume"].array.shape[0] // 2
                    st.success("Volume loaded.")
                except Exception as exc:
                    st.session_state.nii_state = None
                    st.error(str(exc))

        st.markdown('<div class="med-section-label">Session</div>', unsafe_allow_html=True)
        if st.session_state.nii_state is not None and st.button("Reset Mask"):
            reset_mask(st.session_state.nii_state)
            st.info("Mask reset.")

    state = st.session_state.nii_state
    if state is None:
        st.markdown(
            """
            <div class="med-empty">
                <h2>Load a NIfTI volume to start</h2>
                <p>Use the control console to import a .nii.gz study, confirm the model and device, then draw one rectangle on the target slice to run MedSAM2 propagation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    volume = state["volume"]

    max_slice = int(volume.array.shape[0] - 1)
    slice_idx = st.slider(
        "Slice index",
        min_value=0,
        max_value=max_slice,
        value=int(st.session_state.get("slice_idx", max_slice // 2)),
        step=1,
        key="slice_idx",
    )
    st.markdown(
        render_metric_grid(
            case_summary_items(volume, slice_idx, torch_status, config_name)
        ),
        unsafe_allow_html=True,
    )
    mask_stats = mask_summary(state)
    has_mask = mask_stats["voxels"] > 0
    st.markdown(
        f"""
        <div class="med-workflow">
            <div class="med-workflow-step active">1. Review slice</div>
            <div class="med-workflow-step {'active' if True else ''}">2. Draw target box</div>
            <div class="med-workflow-step {'active' if state.get("last_bbox") else ''}">3. Segment volume</div>
            <div class="med-workflow-step {'active' if has_mask else ''}">4. Export mask</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prompt_rgb = image_for_slice(state, slice_idx, with_mask=False)
    overlay_rgb = image_for_slice(state, slice_idx, with_mask=True)
    display_size = compute_display_size(prompt_rgb.shape)
    display_width, display_height = display_size

    prompt_col, overlay_col = st.columns(2)
    with prompt_col:
        st.markdown(
            """
            <div class="med-viewer-title">
                <h2>Prompt Viewer</h2>
            </div>
            <div class="med-viewer-caption">Draw one rectangle around the target. Wait for the box coordinates before running segmentation.</div>
            """,
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

        if st.button("Segment", type="primary", disabled=not torch_status["available"]):
            if bbox is None:
                st.error("Draw a rectangle or enter a valid box first.")
            else:
                with st.spinner("Running MedSAM2 propagation..."):
                    try:
                        output_path = segment_current_slice(
                            state,
                            bbox,
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

    with overlay_col:
        st.markdown(
            """
            <div class="med-viewer-title">
                <h2>Segmentation Overlay</h2>
            </div>
            <div class="med-viewer-caption">Current slice overlay and latest propagated mask statistics.</div>
            """,
            unsafe_allow_html=True,
        )
        st.image(overlay_rgb, caption="Current slice overlay")
        st.markdown(render_mask_summary(mask_stats), unsafe_allow_html=True)
        if state.get("output_path") and os.path.exists(state["output_path"]):
            with open(state["output_path"], "rb") as handle:
                st.download_button(
                    "Download latest mask (.nii.gz)",
                    data=handle,
                    file_name=basename(state["output_path"]),
                    mime="application/gzip",
                )


if __name__ == "__main__":
    main()
