import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import SimpleITK as sitk


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class NiftiVolume:
    path: str
    image: sitk.Image
    array: np.ndarray
    preview: np.ndarray


def validate_nifti_path(path: str) -> None:
    if not path or not str(path).lower().endswith(".nii.gz"):
        raise ValueError("Please upload a .nii.gz NIfTI file.")


def normalize_volume_to_uint8(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    data_min = float(np.min(volume))
    data_max = float(np.max(volume))
    if data_max <= data_min:
        return np.zeros(volume.shape, dtype=np.uint8)
    normalized = (volume - data_min) / (data_max - data_min)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def prepare_video_volume(volume: np.ndarray, image_size: int, device: str):
    import torch
    import torch.nn.functional as F

    volume_tensor = torch.as_tensor(
        volume,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    if volume_tensor.shape[-2:] != (image_size, image_size):
        volume_tensor = F.interpolate(
            volume_tensor,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
        )
    volume_tensor = volume_tensor.div(255.0)
    volume_tensor = volume_tensor.expand(-1, 3, -1, -1).clone()
    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=volume_tensor.dtype,
        device=volume_tensor.device,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD,
        dtype=volume_tensor.dtype,
        device=volume_tensor.device,
    ).view(1, 3, 1, 1)
    volume_tensor.sub_(mean).div_(std)
    return volume_tensor


def load_nifti_volume(path: str) -> NiftiVolume:
    validate_nifti_path(path)
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {array.shape}.")
    preview = normalize_volume_to_uint8(array)
    return NiftiVolume(path=path, image=image, array=array, preview=preview)


def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.max(mask, axis=-1)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D prompt mask, got shape {mask.shape}.")

    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("The prompt mask is empty. Draw over the target before segmenting.")

    return np.array(
        [
            int(np.min(x_indices)),
            int(np.min(y_indices)),
            int(np.max(x_indices)),
            int(np.max(y_indices)),
        ],
        dtype=np.int64,
    )


def overlay_mask_on_slice(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    color: Sequence[int] = (255, 80, 40),
    alpha: float = 0.45,
) -> np.ndarray:
    image_slice = np.asarray(image_slice)
    if image_slice.ndim == 2:
        image_rgb = np.repeat(image_slice[..., None], 3, axis=-1).astype(np.float32)
    elif image_slice.ndim == 3 and image_slice.shape[-1] == 3:
        image_rgb = image_slice.astype(np.float32)
    else:
        raise ValueError(f"Expected a 2D grayscale or RGB slice, got shape {image_slice.shape}.")

    mask_bool = np.asarray(mask_slice) > 0
    if mask_bool.shape != image_rgb.shape[:2]:
        raise ValueError("Mask and image slice shapes do not match.")

    tint = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    image_rgb[mask_bool] = (1.0 - alpha) * image_rgb[mask_bool] + alpha * tint
    return np.clip(image_rgb, 0, 255).astype(np.uint8)


def save_mask_nifti(mask: np.ndarray, source: NiftiVolume, output_path: str) -> str:
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape != source.array.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match source shape {source.array.shape}.")

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    mask_image = sitk.GetImageFromArray(mask)
    mask_image.CopyInformation(source.image)
    sitk.WriteImage(mask_image, output_path)
    return output_path


class NiftiSegmenter:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        image_size: int = 512,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.image_size = image_size
        self._predictor = None

    def _get_predictor(self):
        if self._predictor is None:
            from sam2.build_sam import build_sam2_video_predictor_npz

            self._predictor = build_sam2_video_predictor_npz(
                self.config_path,
                self.checkpoint_path,
                device=self.device,
            )
        return self._predictor

    def segment_with_box(
        self,
        volume: NiftiVolume,
        frame_idx: int,
        bbox: Sequence[int],
        obj_id: int = 1,
        existing_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        import torch

        predictor = self._get_predictor()
        frame_idx = int(frame_idx)
        if frame_idx < 0 or frame_idx >= volume.preview.shape[0]:
            raise ValueError(f"Slice index {frame_idx} is outside volume depth {volume.preview.shape[0]}.")

        bbox = np.asarray(bbox, dtype=np.int64)
        if bbox.shape != (4,):
            raise ValueError("Box prompt must be [x_min, y_min, x_max, y_max].")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("Box prompt must have positive width and height.")

        if existing_mask is None:
            output_mask = np.zeros(volume.preview.shape, dtype=np.uint8)
        else:
            output_mask = np.asarray(existing_mask, dtype=np.uint8).copy()
            if output_mask.shape != volume.preview.shape:
                raise ValueError("Existing mask shape does not match the loaded volume.")

        img_resized = prepare_video_volume(
            volume.preview,
            image_size=self.image_size,
            device=predictor.device,
        )
        video_height, video_width = volume.preview.shape[1:3]

        autocast_device = "cuda" if str(self.device).startswith("cuda") else None
        autocast_context = (
            torch.autocast(autocast_device, dtype=torch.bfloat16)
            if autocast_device is not None
            else nullcontext()
        )

        with torch.inference_mode(), autocast_context:
            inference_state = predictor.init_state(img_resized, video_height, video_width)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=bbox,
            )
            self._propagate_into_mask(predictor, inference_state, output_mask)
            predictor.reset_state(inference_state)

            inference_state = predictor.init_state(img_resized, video_height, video_width)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=bbox,
            )
            self._propagate_into_mask(predictor, inference_state, output_mask, reverse=True)
            predictor.reset_state(inference_state)

        return output_mask.astype(np.uint8)

    @staticmethod
    def _propagate_into_mask(predictor, inference_state, output_mask: np.ndarray, reverse: bool = False) -> None:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            reverse=reverse,
        ):
            for obj_offset, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[obj_offset] > 0.0).cpu().numpy()[0]
                output_mask[int(out_frame_idx), mask] = int(obj_id)
