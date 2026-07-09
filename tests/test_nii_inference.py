import os
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

from nii_inference import (
    NiftiVolume,
    load_nifti_volume,
    mask_to_bbox,
    overlay_mask_on_slice,
    save_mask_nifti,
    validate_nifti_path,
)


def write_test_image(path, array):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.7, 0.8, 2.5))
    image.SetOrigin((11.0, 12.0, 13.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0))
    sitk.WriteImage(image, path)
    return image


class NiftiInferenceTests(unittest.TestCase):
    def test_validate_nifti_path_accepts_nii_gz(self):
        validate_nifti_path("case_001.nii.gz")

    def test_validate_nifti_path_rejects_other_extensions(self):
        with self.assertRaisesRegex(ValueError, "nii.gz"):
            validate_nifti_path("case_001.nii")

    def test_load_nifti_volume_reads_array_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "case.nii.gz")
            source = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
            image = write_test_image(path, source)

            volume = load_nifti_volume(path)

            self.assertIsInstance(volume, NiftiVolume)
            np.testing.assert_array_equal(volume.array, source)
            np.testing.assert_allclose(volume.image.GetSpacing(), image.GetSpacing())
            self.assertEqual(volume.preview.shape, source.shape)
            self.assertEqual(volume.preview.dtype, np.uint8)

    def test_mask_to_bbox_returns_xyxy_from_drawn_mask(self):
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2:6, 3:8] = 255

        bbox = mask_to_bbox(mask)

        np.testing.assert_array_equal(bbox, np.array([3, 2, 7, 5]))

    def test_mask_to_bbox_rejects_empty_mask(self):
        with self.assertRaisesRegex(ValueError, "prompt mask is empty"):
            mask_to_bbox(np.zeros((4, 4), dtype=np.uint8))

    def test_overlay_mask_on_slice_tints_mask_region(self):
        image_slice = np.full((4, 4), 100, dtype=np.uint8)
        mask_slice = np.zeros((4, 4), dtype=np.uint8)
        mask_slice[1:3, 1:3] = 1

        overlay = overlay_mask_on_slice(image_slice, mask_slice)

        self.assertEqual(overlay.shape, (4, 4, 3))
        self.assertTrue(np.any(overlay[1, 1] != overlay[0, 0]))

    def test_save_mask_nifti_preserves_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "case.nii.gz")
            output_path = os.path.join(tmpdir, "case_mask.nii.gz")
            source = np.zeros((2, 3, 4), dtype=np.int16)
            image = write_test_image(input_path, source)
            volume = load_nifti_volume(input_path)
            mask = np.zeros_like(source, dtype=np.uint8)
            mask[:, 1, 2] = 1

            written = save_mask_nifti(mask, volume, output_path)
            output = sitk.ReadImage(written)

            np.testing.assert_allclose(output.GetSpacing(), volume.image.GetSpacing())
            np.testing.assert_allclose(output.GetOrigin(), volume.image.GetOrigin())
            np.testing.assert_allclose(output.GetDirection(), volume.image.GetDirection())
            np.testing.assert_array_equal(sitk.GetArrayFromImage(output), mask)


if __name__ == "__main__":
    unittest.main()
