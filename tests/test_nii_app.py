import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import nii_app
from nii_app import materialize_uploaded_nifti, strip_nii_gz


class FakeUpload:
    def __init__(self, name, payload):
        self.name = name
        self._payload = payload

    def getbuffer(self):
        return memoryview(self._payload)


class NiftiAppTests(unittest.TestCase):
    def setUp(self):
        nii_app.get_torch_status.cache_clear()

    def test_strip_nii_gz_removes_compound_extension(self):
        self.assertEqual(strip_nii_gz("patient_001.nii.gz"), "patient_001")

    def test_materialize_uploaded_nifti_writes_upload_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            upload = FakeUpload("scan.nii.gz", b"nifti-bytes")

            path = materialize_uploaded_nifti(upload, tmpdir)

            self.assertTrue(path.endswith("scan.nii.gz"))
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), b"nifti-bytes")

    def test_materialize_uploaded_nifti_rejects_non_nifti_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            upload = FakeUpload("scan.nii", b"data")

            with self.assertRaisesRegex(ValueError, "nii.gz"):
                materialize_uploaded_nifti(upload, tmpdir)

    def test_torch_import_failure_defaults_to_cpu_and_reports_error(self):
        with patch("nii_app.importlib.import_module", side_effect=OSError("c10.dll failed")):
            status = nii_app.get_torch_status()

        self.assertFalse(status["available"])
        self.assertEqual(status["device"], "cpu")
        self.assertIn("c10.dll failed", status["error"])

    def test_canvas_updates_streamlit_so_drawn_box_can_segment(self):
        self.assertTrue(nii_app.CANVAS_REALTIME_UPDATE)

    def test_select_default_name_prefers_requested_choice(self):
        choices = ["a", "sam2.1_hiera_t512", "c"]

        self.assertEqual(
            nii_app.select_default_name(choices, "sam2.1_hiera_t512"),
            "sam2.1_hiera_t512",
        )

    def test_select_default_name_falls_back_to_first_choice(self):
        self.assertEqual(nii_app.select_default_name(["first", "second"], "missing"), "first")

    def test_case_summary_items_formats_loaded_volume_for_header(self):
        volume = SimpleNamespace(
            path="C:/data/ct.nii.gz",
            array=SimpleNamespace(shape=(162, 512, 512)),
            image=SimpleNamespace(GetSpacing=lambda: (0.8, 0.8, 2.0)),
        )
        torch_status = {"device": "cuda", "available": True}

        items = nii_app.case_summary_items(volume, 81, torch_status, "sam2.1_hiera_t512")

        self.assertEqual(items[0], ("Case", "ct.nii.gz", "Loaded"))
        self.assertIn(("Dimensions", "162 x 512 x 512", "(D x H x W)"), items)
        self.assertIn(("Spacing", "0.8 x 0.8 x 2", "mm"), items)
        self.assertIn(("Voxel Count", "42,467,328", "voxels"), items)
        self.assertIn(("Slice", "81 / 161", "Current axial index"), items)
        self.assertIn(("Device", "CUDA", "Ready"), items)
        self.assertIn(("Model", "sam2.1_hiera_t512", "Checkpoint selected"), items)

    def test_thumbnail_indices_centers_current_slice(self):
        self.assertEqual(nii_app.thumbnail_indices(81, 162), [78, 79, 80, 81, 82, 83, 84])
        self.assertEqual(nii_app.thumbnail_indices(1, 5), [0, 1, 2, 3, 4])

    def test_mask_summary_calculates_volume_voxels_and_coverage(self):
        mask = nii_app.np.zeros((2, 4, 4), dtype=nii_app.np.uint8)
        mask[:, :2, :2] = 1
        state = {
            "mask": mask,
            "volume": SimpleNamespace(image=SimpleNamespace(GetSpacing=lambda: (1.0, 1.0, 2.0))),
        }

        summary = nii_app.mask_summary(state)

        self.assertEqual(summary["voxels"], 8)
        self.assertEqual(summary["volume_cm3"], 0.016)
        self.assertEqual(summary["coverage"], 25.0)


if __name__ == "__main__":
    unittest.main()
