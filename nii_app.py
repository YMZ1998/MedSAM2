import importlib
import os
import shutil
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
from nii_ui import (
    product_css,
    render_brand,
    render_empty_thumbnails,
    render_file_card,
    render_mask_summary,
    render_placeholder_status,
    render_section_label,
    render_status_grid,
    render_viewer_title,
)


SEGMENTER_CACHE = {}
CANVAS_REALTIME_UPDATE = True
DEFAULT_CONFIG_NAME = "sam2.1_hiera_t512"
DEFAULT_CHECKPOINT_NAME = "MedSAM2_latest"
WINDOW_PRESETS = {
    "Auto": None,
    "CT Soft Tissue": (40.0, 400.0),
    "CT Lung": (-600.0, 1500.0),
    "CT Bone": (400.0, 1800.0),
}


def strip_nii_gz(filename):
    name = basename(filename)
    return name[:-7] if name.lower().endswith(".nii.gz") else splitext(name)[0]


def discover_files(folder, pattern):
    return [path for path in sorted(glob(os.path.join(folder, pattern))) if os.path.isfile(path)]


def display_name(path):
    return splitext(basename(path))[0] if path.endswith((".yaml", ".pt")) else basename(path)


def build_file_map(paths):
    return {display_name(path): path for path in paths}


def build_config_map(paths):
    return {
        display_name(path): os.path.join("configs", basename(path)).replace("\\", "/")
        for path in paths
    }


CONFIG_MAP = build_config_map(discover_files(os.path.join("sam2", "configs"), "*.yaml"))
CHECKPOINT_MAP = build_file_map(discover_files("checkpoints", "*.pt"))


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
    with open(output_path, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return output_path


def get_segmenter(config_name, checkpoint_name, device):
    if config_name not in CONFIG_MAP or checkpoint_name not in CHECKPOINT_MAP:
        raise ValueError("The selected model or checkpoint is no longer available.")
    key = (CONFIG_MAP[config_name], CHECKPOINT_MAP[checkpoint_name], device)
    if key not in SEGMENTER_CACHE:
        SEGMENTER_CACHE[key] = NiftiSegmenter(key[0], key[1], device=device)
    return SEGMENTER_CACHE[key]


def format_measurement(value):
    return f"{value:.4g}"


def case_summary_items(volume, slice_idx, torch_status, config_name):
    depth, height, width = volume.array.shape
    spacing = volume.image.GetSpacing()
    return [
        ("Case", basename(volume.path), "Loaded"),
        ("Dimensions", f"{depth} x {height} x {width}", "D x H x W"),
        ("Spacing", " x ".join(format_measurement(v) for v in spacing), "mm"),
        ("Voxel Count", f"{int(np.prod(volume.array.shape)):,}", "voxels"),
        ("Slice", f"{int(slice_idx)} / {depth - 1}", "Axial index"),
        ("Device", torch_status["device"].upper(), "Ready" if torch_status["available"] else "Offline"),
        ("Model", config_name, "Ready"),
    ]


def thumbnail_indices(slice_idx, depth, radius=3):
    depth = int(depth)
    if depth <= 0:
        return []
    window = min(depth, radius * 2 + 1)
    start = max(0, min(int(slice_idx) - radius, depth - window))
    return list(range(start, start + window))


def mask_summary(state):
    voxels = int(np.count_nonzero(state["mask"]))
    spacing = state["volume"].image.GetSpacing()
    return {
        "voxels": voxels,
        "volume_cm3": round(voxels * float(np.prod(spacing)) / 1000.0, 3),
        "coverage": round(voxels / float(state["mask"].size) * 100.0, 2),
    }


def infer_auto_window(array):
    flat = np.asarray(array).reshape(-1)
    stride = max(1, flat.size // 1_000_000)
    low, high = np.percentile(flat[::stride], (1.0, 99.0))
    width = max(1.0, float(high - low))
    return float((low + high) / 2.0), width


def window_slice_to_uint8(image_slice, center, width):
    width = max(float(width), 1.0)
    low = float(center) - width / 2.0
    scaled = (np.asarray(image_slice, dtype=np.float32) - low) / width
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def create_session_state(volume):
    center, width = infer_auto_window(volume.array)
    return {
        "volume": volume,
        "mask": np.zeros(volume.array.shape, dtype=np.uint8),
        "output_dir": tempfile.mkdtemp(prefix="medsam2_nii_"),
        "output_path": None,
        "last_bbox": None,
        "window_center": center,
        "window_width": width,
        "auto_window": (center, width),
        "window_preset": "Auto",
    }


def cleanup_state(state):
    if state and state.get("output_dir"):
        shutil.rmtree(state["output_dir"], ignore_errors=True)


def image_for_slice(state, slice_idx, with_mask=True, alpha=0.5, show_mask=True):
    gray = window_slice_to_uint8(
        state["volume"].array[int(slice_idx)],
        state["window_center"],
        state["window_width"],
    )
    if with_mask and show_mask:
        return overlay_mask_on_slice(gray, state["mask"][int(slice_idx)], color=(16, 185, 189), alpha=alpha)
    return np.repeat(gray[..., None], 3, axis=-1)


def compute_display_size(image_shape, max_width=720):
    height, width = image_shape[:2]
    scale = min(1.0, max_width / float(width))
    return max(1, int(width * scale)), max(1, int(height * scale))


def bbox_from_canvas(canvas_json, display_size, source_shape):
    objects = (canvas_json or {}).get("objects") or []
    if not objects:
        raise ValueError("Draw a rectangle over the target before segmenting.")
    item = objects[-1]
    if item.get("type") != "rect":
        raise ValueError("Use the rectangle tool to mark the target.")
    display_width, display_height = display_size
    source_height, source_width = source_shape[:2]
    left, top = float(item.get("left", 0)), float(item.get("top", 0))
    width = float(item.get("width", 0)) * float(item.get("scaleX", 1))
    height = float(item.get("height", 0)) * float(item.get("scaleY", 1))
    coords = np.array(
        [
            round(left * source_width / display_width),
            round(top * source_height / display_height),
            round((left + width) * source_width / display_width),
            round((top + height) * source_height / display_height),
        ],
        dtype=np.int64,
    )
    coords[[0, 2]] = np.clip(coords[[0, 2]], 0, source_width - 1)
    coords[[1, 3]] = np.clip(coords[[1, 3]], 0, source_height - 1)
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise ValueError("The rectangle prompt is too small.")
    return coords


def segment_current_slice(state, bbox, slice_idx, config_name, checkpoint_name, device):
    torch_status = get_torch_status()
    if not torch_status["available"]:
        raise RuntimeError(f"PyTorch is unavailable in this environment: {torch_status['error']}")
    mask = get_segmenter(config_name, checkpoint_name, device).segment_with_box(
        state["volume"], int(slice_idx), bbox
    )
    output_path = os.path.join(state["output_dir"], f"{strip_nii_gz(state['volume'].path)}_mask.nii.gz")
    save_mask_nifti(mask, state["volume"], output_path)
    state.update(mask=mask, output_path=output_path, last_bbox=[int(v) for v in bbox])
    return output_path


def reset_mask(state):
    state["mask"].fill(0)
    state["output_path"] = None
    state["last_bbox"] = None


def get_streamlit_canvas():
    try:
        from streamlit_drawable_canvas import st_canvas
        return st_canvas
    except Exception:
        return None


def set_slice(st, value, max_slice):
    st.session_state.slice_idx = max(0, min(int(value), int(max_slice)))
    st.session_state.current_bbox = None
    st.session_state.bbox_slice = None


def render_slice_toolbar(st, max_slice):
    previous, slider, next_button = st.columns([0.08, 0.84, 0.08])
    with previous:
        if st.button("‹", key="slice_previous", help="Previous slice", disabled=st.session_state.slice_idx <= 0):
            set_slice(st, st.session_state.slice_idx - 1, max_slice)
            st.rerun()
    with slider:
        st.slider("Slice", 0, max_slice, key="slice_idx", label_visibility="collapsed")
    with next_button:
        if st.button("›", key="slice_next", help="Next slice", disabled=st.session_state.slice_idx >= max_slice):
            set_slice(st, st.session_state.slice_idx + 1, max_slice)
            st.rerun()


def render_thumbnails(st, state, active_idx):
    indices = thumbnail_indices(active_idx, state["volume"].array.shape[0])
    columns = st.columns(len(indices))
    for column, idx in zip(columns, indices):
        with column:
            st.image(image_for_slice(state, idx, with_mask=False), use_column_width=True)
            label = f"● {idx}" if idx == active_idx else str(idx)
            if st.button(label, key=f"thumb_{idx}"):
                set_slice(st, idx, state["volume"].array.shape[0] - 1)
                st.rerun()


def render_empty_workspace(st, model_name, device):
    st.markdown(render_placeholder_status(model_name, device), unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown(render_viewer_title(1, "Prompt (Draw Rectangle)"), unsafe_allow_html=True)
        st.markdown('<div class="viewer-placeholder">Load a .nii.gz volume to begin</div>', unsafe_allow_html=True)
        st.markdown(render_empty_thumbnails(), unsafe_allow_html=True)
    with right:
        st.markdown(render_viewer_title(2, "Segmentation Overlay"), unsafe_allow_html=True)
        st.markdown('<div class="viewer-placeholder">The generated mask will appear here</div>', unsafe_allow_html=True)
        st.markdown(render_mask_summary({"voxels": 0, "volume_cm3": 0, "coverage": 0}), unsafe_allow_html=True)
    st.markdown('<div class="info-strip">Import a .nii.gz study, choose a slice, draw one rectangle, then run segmentation.</div>', unsafe_allow_html=True)


def main():
    import streamlit as st

    st.set_page_config(page_title="MedSAM2 NIfTI Studio", page_icon="+", layout="wide", initial_sidebar_state="expanded")
    st.markdown(product_css(), unsafe_allow_html=True)
    if not CONFIG_MAP or not CHECKPOINT_MAP:
        st.error("Model configuration or checkpoint files are missing.")
        st.stop()

    defaults = {
        "nii_state": None,
        "current_bbox": None,
        "bbox_slice": None,
        "segment_requested": False,
        "overlay_alpha": 0.5,
        "show_mask": True,
        "window_preset_input": "Auto",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "upload_workspace" not in st.session_state:
        st.session_state.upload_workspace = tempfile.mkdtemp(prefix="medsam2_upload_")

    torch_status = get_torch_status()
    config_choices, checkpoint_choices = list(CONFIG_MAP), list(CHECKPOINT_MAP)
    with st.sidebar:
        st.markdown(render_brand(), unsafe_allow_html=True)
        st.markdown(render_section_label("Upload & Data"), unsafe_allow_html=True)
        uploaded_file = st.file_uploader("NIfTI file (.nii.gz)", type=["gz"])
        st.markdown(
            render_file_card(st.session_state.get("loaded_upload_name"), st.session_state.get("loaded_upload_size", 0)),
            unsafe_allow_html=True,
        )

        st.markdown(render_section_label("Model & Checkpoint"), unsafe_allow_html=True)
        config_name = st.selectbox("Model", config_choices, index=config_choices.index(select_default_name(config_choices, DEFAULT_CONFIG_NAME)))
        checkpoint_name = st.selectbox("Checkpoint", checkpoint_choices, index=checkpoint_choices.index(select_default_name(checkpoint_choices, DEFAULT_CHECKPOINT_NAME)))
        st.markdown(render_section_label("Device"), unsafe_allow_html=True)
        device = st.selectbox("Device", ["cuda", "cpu"], index=0 if torch_status["device"] == "cuda" else 1, disabled=not torch_status["available"])
        runtime_class = "runtime" if torch_status["available"] else "runtime offline"
        runtime_text = (
            f"&#9679;&nbsp; PyTorch {torch_status['version']}<br>Default device: {torch_status['device']}"
            if torch_status["available"] else "PyTorch unavailable; segmentation is disabled"
        )
        st.markdown(f'<div class="{runtime_class}">{runtime_text}</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            upload_id = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
            if upload_id != st.session_state.get("loaded_upload_id"):
                try:
                    input_path = materialize_uploaded_nifti(uploaded_file, st.session_state.upload_workspace)
                    new_state = create_session_state(load_nifti_volume(input_path))
                    cleanup_state(st.session_state.nii_state)
                    st.session_state.nii_state = new_state
                    st.session_state.loaded_upload_id = upload_id
                    st.session_state.loaded_upload_name = uploaded_file.name
                    st.session_state.loaded_upload_size = getattr(uploaded_file, "size", 0)
                    st.session_state.slice_idx = new_state["volume"].array.shape[0] // 2
                    st.session_state.current_bbox = None
                    st.session_state.bbox_slice = None
                    st.session_state.window_preset_input = "Auto"
                    st.session_state.window_level_input = new_state["window_center"]
                    st.session_state.window_width_input = new_state["window_width"]
                    st.session_state.flash_message = "Volume loaded successfully."
                    st.rerun()
                except Exception as exc:
                    st.error(f"Unable to load volume: {exc}")

        state = st.session_state.nii_state
        with st.expander("Display settings", expanded=False):
            preset = st.selectbox(
                "Window preset",
                list(WINDOW_PRESETS),
                key="window_preset_input",
                disabled=state is None,
            )
            if state is not None:
                if state["window_preset"] != preset:
                    center, width = state["auto_window"] if WINDOW_PRESETS[preset] is None else WINDOW_PRESETS[preset]
                    state["window_preset"] = preset
                    st.session_state.window_level_input = center
                    st.session_state.window_width_input = width
                state["window_center"] = st.number_input(
                    "Window level", step=10.0, key="window_level_input"
                )
                state["window_width"] = st.number_input(
                    "Window width", min_value=1.0, step=10.0, key="window_width_input"
                )
            st.session_state.show_mask = st.checkbox("Show mask overlay", value=st.session_state.show_mask)
            st.session_state.overlay_alpha = st.slider("Overlay opacity", 0.1, 0.9, st.session_state.overlay_alpha, 0.05)

        st.markdown(render_section_label("Actions"), unsafe_allow_html=True)
        can_segment = state is not None and torch_status["available"]
        if st.button("Run segmentation", type="primary", disabled=not can_segment):
            st.session_state.segment_requested = True
        if st.button("Reset mask", disabled=state is None):
            reset_mask(state)
            st.session_state.current_bbox = None
            st.session_state.bbox_slice = None
            st.session_state.flash_message = "Mask reset."
            st.rerun()
        output_path = state.get("output_path") if state else None
        if output_path and os.path.exists(output_path):
            with open(output_path, "rb") as handle:
                st.download_button("Download mask", handle, file_name=basename(output_path), mime="application/gzip")
        else:
            st.download_button("Download mask", b"", file_name="mask.nii.gz", disabled=True)

    state = st.session_state.nii_state
    if message := st.session_state.pop("flash_message", None):
        st.success(message)
    if state is None:
        render_empty_workspace(st, config_name, torch_status["device"])
        return

    max_slice = state["volume"].array.shape[0] - 1
    st.markdown(render_status_grid(case_summary_items(state["volume"], st.session_state.slice_idx, torch_status, config_name)), unsafe_allow_html=True)
    render_slice_toolbar(st, max_slice)
    slice_idx = int(st.session_state.slice_idx)
    prompt_rgb = image_for_slice(state, slice_idx, with_mask=False)
    overlay_rgb = image_for_slice(state, slice_idx, alpha=st.session_state.overlay_alpha, show_mask=st.session_state.show_mask)
    display_size = compute_display_size(prompt_rgb.shape)

    prompt_col, overlay_col = st.columns(2)
    with prompt_col:
        st.markdown(render_viewer_title(1, "Prompt (Draw Rectangle)"), unsafe_allow_html=True)
        st_canvas = get_streamlit_canvas()
        bbox = None
        if st_canvas is not None:
            canvas_result = st_canvas(
                fill_color="rgba(242, 154, 56, 0.22)", stroke_width=2, stroke_color="#f29a38",
                background_image=Image.fromarray(prompt_rgb), drawing_mode="rect",
                width=display_size[0], height=display_size[1], update_streamlit=CANVAS_REALTIME_UPDATE,
                key=f"canvas_{slice_idx}",
            )
            if canvas_result.json_data:
                try:
                    bbox = bbox_from_canvas(canvas_result.json_data, display_size, prompt_rgb.shape)
                    st.session_state.current_bbox = bbox
                    st.session_state.bbox_slice = slice_idx
                    st.markdown(f'<div class="note">Prompt ready: {bbox.tolist()}</div>', unsafe_allow_html=True)
                except ValueError as exc:
                    st.markdown(f'<div class="note warn">{exc}</div>', unsafe_allow_html=True)
        else:
            st.image(prompt_rgb, use_column_width=True)
            st.error("Rectangle drawing component is unavailable. Reinstall the interactive-demo dependencies.")

        if st.session_state.segment_requested:
            st.session_state.segment_requested = False
            active_bbox = bbox if bbox is not None else st.session_state.current_bbox
            if active_bbox is None or st.session_state.bbox_slice != slice_idx:
                st.error("Draw a rectangle on the current slice before running segmentation.")
            else:
                try:
                    with st.spinner("MedSAM2 is propagating the mask through the volume..."):
                        segment_current_slice(state, active_bbox, slice_idx, config_name, checkpoint_name, device)
                    st.session_state.flash_message = "Segmentation completed. The mask is ready to review and download."
                    st.rerun()
                except Exception as exc:
                    st.error(f"Segmentation failed: {exc}")
        render_thumbnails(st, state, slice_idx)

    with overlay_col:
        st.markdown(render_viewer_title(2, "Segmentation Overlay"), unsafe_allow_html=True)
        st.image(overlay_rgb, use_column_width=True)
        st.markdown(render_mask_summary(mask_summary(state)), unsafe_allow_html=True)
        mask_state = "visible" if st.session_state.show_mask else "hidden"
        st.markdown(
            f'<div class="quick-card"><div class="card-title">Review controls</div>'
            f'<span style="display:inline-block;width:28px;height:28px;border-radius:5px;background:#10b9bd;vertical-align:middle;margin-right:10px"></span>'
            f'<span style="color:#52647e;font-weight:720">Overlay {mask_state} &nbsp; {int(st.session_state.overlay_alpha * 100)}%</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="info-strip">Draw one rectangle around the target on the prompt viewer. Review the propagated mask on adjacent slices before downloading.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
