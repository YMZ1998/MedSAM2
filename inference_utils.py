import numpy as np
import torch
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_volume_to_uint8(volume):
    volume = np.asarray(volume, dtype=np.float32)
    data_min = float(volume.min())
    data_max = float(volume.max())
    if data_max <= data_min:
        return np.zeros(volume.shape, dtype=np.uint8)
    normalized = (volume - data_min) / (data_max - data_min)
    return (normalized * 255.0).astype(np.uint8)


def prepare_video_volume(volume, image_size, device):
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
