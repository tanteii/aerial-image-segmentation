# Aerial Image Segmentation
Pixel-level semantic segmentation of aerial RGB imagery using traditional machine learning. Built as part of a computer vision project at UNSW. Extended for further evaluation of machine learning and computer vision methods.

## Overview

This project classifies individual pixels in aerial canopy images as either a dead tree or healty tree via a supervised ML pipeline. The pipeline is built on the [DetecTree](https://github.com/martibosch/detectree) framework, relying on handcrafted pixel-level features such as colour, texture and entropy descriptors that are fed into a LightGBM gradient boosting classifier.

Beyond the DetecTree implementation, the pipeline was extended with validation-aware training with early stopping, per-epoch IoU tracking and training curve visualisation.

## Pipeline

**Data preparation**
- Stratified train / validation / test split (80 / 10 / 10)
- 8-way geometric augmentation on train and val sets: horizontal flip, 
  vertical flip, combined flip, transpose, and 90/180/270° rotations
- Resize all images to 256×256
- Convert PNG ground truth masks to GeoTIFF response tiles for 
  DetecTree compatibility

**Training**
- Extended `ClassifierTrainer` with a custom subclass to support 
  validation data, LightGBM early stopping, and callbacks for per-round 
  IoU
- Saves trained model to `.joblib` for reuse

**Evaluation**
- Mean Intersection over Union (IoU) across the full test set
- Per-image IoU scatter plot against the mean
- Side-by-side visualisation of RGB input, ground truth mask, and 
  predicted mask


## Requirements

```bash
pip install pandas joblib pillow scikit-learn torchvision numpy \
            rasterio lightgbm matplotlib opencv-python PyMaxflow detectree
```

## Usage

Recommended: run the notebook in an empty directory — the pipeline 
creates several output directories during execution.

Run all cells top to bottom. Set `save_model = True` to persist the 
trained classifier to `detectree_classifier.joblib`.

Use the final notebook section with a pre-trained `.joblib` file. 
Point `model_path` at your saved classifier and run.

## Dataset

The dataset used is not included in this repository.

Notebook expects aerial RGB images and corresponding binary masks organised as:

USA_segmentation/
RGB_images/   # input tiles, named rgb_<id>.png
masks/        # ground truth masks, named mask_<id>.png

## References

- [DetecTree](https://github.com/martibosch/detectree)
- [DetecTree documentation](https://detectree.readthedocs.io/en/latest/)
- [LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/)
