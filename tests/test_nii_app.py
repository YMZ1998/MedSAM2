import io
import os
import tempfile
import unittest
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
        self.assertEqual(nii_app.default_device(), "cpu")

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


if __name__ == "__main__":
    unittest.main()
