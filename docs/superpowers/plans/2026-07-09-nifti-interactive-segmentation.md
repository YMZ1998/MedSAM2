# NIfTI Interactive Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dedicated Gradio web page for interactive MedSAM2 segmentation of uploaded `.nii.gz` volumes.

**Architecture:** Add a focused `nii_inference.py` backend for NIfTI IO, preview generation, prompt conversion, model invocation, overlay rendering, and mask export. Add `nii_app.py` as a separate Gradio entry point so the existing video `app.py` remains unchanged.

**Tech Stack:** Python, SimpleITK, NumPy, PyTorch, Gradio 3.38-style APIs, MedSAM2 `build_sam2_video_predictor_npz`.

---

### Task 1: Testable NIfTI Backend Helpers

**Files:**
- Create: `tests/test_nii_inference.py`
- Create: `nii_inference.py`

- [ ] **Step 1: Write failing tests**

Create tests for `.nii.gz` validation, loading metadata, bbox extraction, overlay rendering, and mask export using small synthetic SimpleITK images.

- [ ] **Step 2: Verify tests fail**

Run: `python -m unittest tests.test_nii_inference -v`

Expected: import failure because `nii_inference.py` does not exist yet.

- [ ] **Step 3: Implement backend helpers**

Create `NiftiVolume`, `validate_nifti_path`, `load_nifti_volume`, `mask_to_bbox`, `overlay_mask_on_slice`, and `save_mask_nifti`.

- [ ] **Step 4: Verify tests pass**

Run: `python -m unittest tests.test_nii_inference -v`

Expected: all backend helper tests pass.

### Task 2: Model Runner Wrapper

**Files:**
- Modify: `nii_inference.py`

- [ ] **Step 1: Extend tests or smoke-check imports**

The model path is GPU and checkpoint dependent, so keep unit tests focused on pure helpers and make the model runner import lazy.

- [ ] **Step 2: Implement `NiftiSegmenter`**

Add a class that lazily builds `build_sam2_video_predictor_npz`, prepares the volume with `prepare_video_volume`, runs forward and reverse propagation from a box prompt, and returns a binary `(D, H, W)` mask.

- [ ] **Step 3: Verify syntax and helper tests**

Run: `python -m py_compile nii_inference.py`
Run: `python -m unittest tests.test_nii_inference -v`

Expected: compile succeeds and tests pass.

### Task 3: Dedicated Gradio App

**Files:**
- Create: `nii_app.py`

- [ ] **Step 1: Implement UI callbacks**

Add upload, slice browsing, sketch-to-box prompting, segmentation, overlay preview, reset, config/checkpoint selection, and mask download.

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile nii_app.py`

Expected: compile succeeds.

### Task 4: Documentation and Final Verification

**Files:**
- Modify: `README.md` only if needed.

- [ ] **Step 1: Verify changed Python files**

Run: `python -m unittest tests.test_nii_inference -v`
Run: `python -m py_compile nii_inference.py nii_app.py`

Expected: tests pass and compile succeeds.

- [ ] **Step 2: Report limitations**

If no GPU/checkpoint smoke test was run, state that clearly and provide the launch command `python nii_app.py`.
