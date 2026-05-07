# Repository Guidelines

## Project Structure & Module Organization

This repository implements a PyTorch reproduction of MultiFormer for single-person WiFi pose estimation. Core modules live at the repository root: `model.py` defines the MultiFormer architecture and loss, `pose_targets.py` generates PCM/PAF targets from COCO-17 keypoints, `dataloader.py` loads HDF5 pose data, `metrics.py` contains evaluation helpers, and `train.py` runs training and validation. Dataset conversion utilities are in `scripts/`, especially `scripts/build_h5_dataset.py`. `MultiFormer.pdf` is the source paper. Generated files such as `__pycache__/`, processed datasets, and checkpoints should stay out of source control.

## Build, Test, and Development Commands

- `python -m py_compile train.py metrics.py model.py dataloader.py pose_targets.py scripts\build_h5_dataset.py`: syntax-check the main code paths.
- `python scripts\build_h5_dataset.py --output-path data\mmfi_pose.h5`: build an HDF5 dataset from the configured raw MMFi/WiFiPose data.
- `python dataloader.py --dataset-root data\mmfi_pose.h5 --preview`: inspect dataset loading and tensor shapes.
- `python train.py --dataset-root data\mmfi_pose.h5 --output-dir runs\multiformer`: train MultiFormer and write checkpoints/metrics under `runs/`.

Use Windows PowerShell paths in local commands unless documenting cross-platform usage.

## Coding Style & Naming Conventions

Use Python 3.10+ style with 4-space indentation, clear type hints, and short docstrings for public helpers. Prefer descriptive tensor names such as `csi_amplitude`, `target_pcm`, `target_paf`, and `predictions`. Keep tensor shape comments exact and update them whenever model or data dimensions change. Follow existing PyTorch module patterns: small `nn.Module` classes, explicit forward inputs, and no hidden global training state.

## Testing Guidelines

There is no formal test suite in this checkout. At minimum, run the compile command above after edits. For model or data changes, add smoke checks that instantiate the dataloader, generate PCM/PAF targets, run a forward pass, compute loss, and verify finite gradients. If adding automated tests, place them under `tests/` and name files `test_*.py`; prefer small CPU-compatible tests.

## Commit & Pull Request Guidelines

This checkout has no Git history, so use concise imperative commit messages such as `Add pose target generation` or `Fix CSI amplitude tokenization`. Pull requests should include a short problem statement, implementation summary, commands run, dataset assumptions, and any changes to tensor shapes or training behavior. Include metrics or logs for training-related changes.

## Security & Configuration Tips

Do not commit raw datasets, HDF5 files, checkpoints, or local paths containing private user information. Keep large artifacts in `data/` or `runs/` and document required paths through command-line arguments rather than hard-coded constants.
