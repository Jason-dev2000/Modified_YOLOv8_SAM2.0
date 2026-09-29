# Modified_YOLOv8 + SAM2.0 Brightfield Cell Detection and Segmentation

Detection and segmentation of cells in brightfield microscopy images, combining a modified **Ultralytics YOLO** detector with **Segment Anything 2.0 (SAM2.0)**.

## Algorithm pipeline

1. **Detection** — a modified YOLOv8 detector, where the original `C2f` module is replaced with a custom **`Modified_YOLOv8`** module, predicts a bounding box and a class label for every cell in the image.

2. **Segmentation** — each detected box is passed to SAM2.0, which produces a set of candidate masks. The highest-confidence mask is kept, and the cell centroid is computed from it.

3. **Post-processing** — optionally localizes a reference microsphere ("Sphere_Probe") by template matching, excludes detections near the microsphere center, suppresses duplicate detections, and then renders dashed contours / centroid cross markers / a scale bar onto the result image.

4. **Output** — the annotated image is saved, and (optionally) a CSV of per-cell coordinates relative to the probe is written.

## Model modifications

- **`Modified_YOLOv8` module**: a custom module defined in `ultralytics/nn/extra_modules/block.py`, which replaces the original `C2f` block in the YOLOv8 backbone.
- Registered in `ultralytics/nn/tasks.py` (`parse_model`) so it can be used directly from a YAML config.
- Model config: `ultralytics/cfg/models/v8/yolov8-Modified_YOLOv8.yaml`.

## Repository layout

```text
.
├── YOLOv8+SAM2.0.py             # detection + SAM2.0 inference script
├── yolov8_sam2.py               # SAM2.0 integration
├── train.py                     # training entry point
├── detect.py                    # detection script
├── val.py                       # validation script
├── track.py                     # tracking script
├── data.yaml                    # dataset config
├── ultralytics/                 # modified Ultralytics source
├── pyproject.toml
├── setup.py
└── .gitignore
```

## Installation

Conda (recommended, reproduces the original environment):

```bash
conda env create -f environment.yml
conda activate myyolo
```

Or pip:

```bash
pip install -r requirements.txt
```

This project modifies the Ultralytics source (adds `Modified_YOLOv8` under `ultralytics/nn/extra_modules/`), so the scripts import the vendored copy in `ultralytics/` rather than the pip-installed package.

## Usage

All paths and parameters are centralized in the config class at the top of each script.

1. Open `YOLOv8+SAM2.0.py` (or `yolov8_sam2.py`) and edit the config values:

   - `input_image_dir` / `input_image_path` — where the input images are.
   - `result_image_dir` / `result_data_dir` — where results are written.
   - `yolo_weights_path` — path to a trained YOLO checkpoint (`best.pt`).
   - `sam_checkpoint` — path to the SAM2.0 checkpoint.

2. Batch mode (process every image in a directory):

   ```bash
   python "YOLOv8+SAM2.0.py"
   ```

3. Training the modified detector:

   ```bash
   python train.py
   ```

## Dataset

The detector was trained on a **self-built dataset** of brightfield microscopy images, annotated with cell bounding boxes. Dataset images and labels are not included in this repository.

## Notes

- Several scripts still contain machine-specific absolute paths (e.g. `D:/Project/YOLOwithSAM/ultralytics-main` and the `runs/.../best.pt` weight paths). Adjust them to your own directory layout before running on another machine.
- Model weights (`*.pt`, `*.pth`) are not included in this repository. Download YOLOv8 pretrained weights and SAM2.0 checkpoints separately.

## License

This project is released under the **AGPL-3.0** license, in accordance with the original Ultralytics YOLO license.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Segment Anything 2.0](https://github.com/facebookresearch/sam2)
