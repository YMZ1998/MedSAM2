import os
import tempfile
from glob import glob
from os.path import basename, splitext

import numpy as np

from nii_inference import (
    NiftiSegmenter,
    load_nifti_volume,
    mask_to_bbox,
    overlay_mask_on_slice,
    save_mask_nifti,
)


SEGMENTER_CACHE = {}


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


def default_device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cuda"


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
        f"Loaded `{basename(volume.path)}` | "
        f"shape `(D, H, W)={tuple(volume.array.shape)}` | "
        f"spacing `({spacing})`"
    )


def image_for_slice(state, slice_idx, with_mask=True):
    volume = state["volume"]
    slice_idx = int(slice_idx)
    image_slice = volume.preview[slice_idx]
    if with_mask:
        return overlay_mask_on_slice(image_slice, state["mask"][slice_idx])
    return np.repeat(image_slice[..., None], 3, axis=-1)


def load_volume(input_file):
    if input_file is None:
        return None, None, None, 0, "Upload a `.nii.gz` file to begin.", None

    input_path = input_file if isinstance(input_file, str) else input_file.name
    volume = load_nifti_volume(input_path)
    depth = int(volume.array.shape[0])
    slice_idx = depth // 2
    state = {
        "volume": volume,
        "mask": np.zeros(volume.array.shape, dtype=np.uint8),
        "output_dir": tempfile.mkdtemp(prefix="medsam2_nii_"),
        "output_path": None,
        "last_bbox": None,
    }
    preview = image_for_slice(state, slice_idx, with_mask=True)
    prompt_image = image_for_slice(state, slice_idx, with_mask=False)
    return (
        state,
        prompt_image,
        preview,
        slice_idx,
        volume_info(volume),
        None,
    )


def update_slice(state, slice_idx):
    if state is None:
        return None, None, "Upload a `.nii.gz` file first."
    return (
        image_for_slice(state, slice_idx, with_mask=False),
        image_for_slice(state, slice_idx, with_mask=True),
        volume_info(state["volume"]),
    )


def extract_drawn_mask(drawing_board):
    if drawing_board is None:
        raise ValueError("Draw a prompt before segmenting.")
    if isinstance(drawing_board, dict):
        mask = drawing_board.get("mask")
    else:
        mask = None
    if mask is None:
        raise ValueError("Draw a prompt before segmenting.")
    return mask


def segment_current_slice(state, drawing_board, slice_idx, config_name, checkpoint_name, device):
    if state is None:
        return None, None, None, "Upload a `.nii.gz` file first."

    try:
        prompt_mask = extract_drawn_mask(drawing_board)
        bbox = mask_to_bbox(prompt_mask)
        segmenter = get_segmenter(config_name, checkpoint_name, device)
        volume = state["volume"]
        mask = segmenter.segment_with_box(volume, int(slice_idx), bbox)
        output_name = f"{strip_nii_gz(volume.path)}_mask.nii.gz"
        output_path = os.path.join(state["output_dir"], output_name)
        save_mask_nifti(mask, volume, output_path)

        state["mask"] = mask
        state["output_path"] = output_path
        state["last_bbox"] = bbox.tolist()
        status = (
            f"Segmentation complete on slice `{int(slice_idx)}` "
            f"with box `{state['last_bbox']}`."
        )
        return state, image_for_slice(state, slice_idx, with_mask=True), output_path, status
    except Exception as exc:
        return state, image_for_slice(state, slice_idx, with_mask=True), state.get("output_path"), f"Error: {exc}"


def reset_mask(state, slice_idx):
    if state is None:
        return None, None, None, "Upload a `.nii.gz` file first."
    state["mask"] = np.zeros(state["volume"].array.shape, dtype=np.uint8)
    state["output_path"] = None
    state["last_bbox"] = None
    return (
        state,
        image_for_slice(state, slice_idx, with_mask=False),
        image_for_slice(state, slice_idx, with_mask=True),
        "Mask reset.",
    )


def build_app():
    import gradio as gr

    if not CONFIG_MAP:
        raise RuntimeError("No config files found under sam2/configs.")
    if not CHECKPOINT_MAP:
        raise RuntimeError("No checkpoint files found under checkpoints.")

    config_choices = list(CONFIG_MAP.keys())
    checkpoint_choices = list(CHECKPOINT_MAP.keys())
    css = """
    #nii_prompt img, #nii_preview img { image-rendering: auto; }
    .status-text { font-size: 0.95rem; }
    """

    with gr.Blocks(css=css) as app:
        gr.Markdown(
            """
            # MedSAM2 NIfTI Interactive Segmentation

            Upload a `.nii.gz` volume, choose a slice, draw over the target, and run 3D propagation.
            """
        )

        state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="Input NIfTI (.nii.gz)",
                    file_types=[".nii.gz"],
                )
                config_dropdown = gr.Dropdown(
                    choices=config_choices,
                    value=config_choices[0],
                    label="Config",
                )
                checkpoint_dropdown = gr.Dropdown(
                    choices=checkpoint_choices,
                    value=checkpoint_choices[0],
                    label="Checkpoint",
                )
                device_dropdown = gr.Dropdown(
                    choices=["cuda", "cpu"],
                    value=default_device(),
                    label="Device",
                )
                slice_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=1,
                    value=0,
                    label="Slice index",
                    interactive=True,
                )
                with gr.Row():
                    segment_button = gr.Button("Segment", variant="primary")
                    reset_button = gr.Button("Reset Mask")
                output_file = gr.File(label="Predicted mask (.nii.gz)")
                status = gr.Markdown("Upload a `.nii.gz` file to begin.")

            with gr.Column(scale=1):
                prompt_image = gr.Image(
                    label="Prompt slice",
                    tool="sketch",
                    type="numpy",
                    elem_id="nii_prompt",
                )
                preview_image = gr.Image(
                    label="Mask overlay",
                    type="numpy",
                    elem_id="nii_preview",
                )

        def handle_load(input_file):
            loaded_state, prompt, preview, slice_idx, message, output_path = load_volume(input_file)
            max_slice = 1
            if loaded_state is not None:
                max_slice = max(0, loaded_state["volume"].array.shape[0] - 1)
            return (
                loaded_state,
                prompt,
                preview,
                gr.Slider.update(maximum=max_slice, value=slice_idx),
                message,
                output_path,
            )

        input_file.change(
            fn=handle_load,
            inputs=[input_file],
            outputs=[state, prompt_image, preview_image, slice_slider, status, output_file],
        )
        slice_slider.release(
            fn=update_slice,
            inputs=[state, slice_slider],
            outputs=[prompt_image, preview_image, status],
        )
        segment_button.click(
            fn=segment_current_slice,
            inputs=[
                state,
                prompt_image,
                slice_slider,
                config_dropdown,
                checkpoint_dropdown,
                device_dropdown,
            ],
            outputs=[state, preview_image, output_file, status],
        )
        reset_button.click(
            fn=reset_mask,
            inputs=[state, slice_slider],
            outputs=[state, prompt_image, preview_image, status],
        )

    return app


if __name__ == "__main__":
    build_app().queue(concurrency_count=1).launch(
        share=False,
        server_name="127.0.0.1",
        server_port=18863,
    )
