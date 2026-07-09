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

    st.set_page_config(page_title="MedSAM2 NIfTI Segmentation", layout="wide")
    st.title("MedSAM2 NIfTI Interactive Segmentation")

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
    if torch_status["available"]:
        st.sidebar.success(
            f"PyTorch {torch_status['version']} | default device: {torch_status['device']}"
        )
    else:
        st.sidebar.error("PyTorch failed to load. Segmentation is disabled in this Python environment.")
        with st.sidebar.expander("PyTorch error"):
            st.code(torch_status["error"])

    with st.sidebar:
        uploaded_file = st.file_uploader("Input NIfTI (.nii.gz)")
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

        if st.session_state.nii_state is not None and st.button("Reset Mask"):
            reset_mask(st.session_state.nii_state)
            st.info("Mask reset.")

    state = st.session_state.nii_state
    if state is None:
        st.info("Upload a .nii.gz volume from the sidebar to begin.")
        return

    volume = state["volume"]
    st.caption(volume_info(volume))

    max_slice = int(volume.array.shape[0] - 1)
    slice_idx = st.slider(
        "Slice index",
        min_value=0,
        max_value=max_slice,
        value=int(st.session_state.get("slice_idx", max_slice // 2)),
        step=1,
        key="slice_idx",
    )

    prompt_rgb = image_for_slice(state, slice_idx, with_mask=False)
    overlay_rgb = image_for_slice(state, slice_idx, with_mask=True)
    display_size = compute_display_size(prompt_rgb.shape)
    display_width, display_height = display_size

    prompt_col, overlay_col = st.columns(2)
    with prompt_col:
        st.subheader("Prompt")
        st.caption("Draw one rectangle around the target. Wait for Current box to appear before clicking Segment.")
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
                    st.caption(f"Current box: {bbox.tolist()}")
                except ValueError as exc:
                    st.warning(str(exc))
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
        st.subheader("Mask overlay")
        st.image(overlay_rgb, caption="Current slice overlay")
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
