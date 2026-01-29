# BVRC: Bias-Variance Risk-Controlled Correction

A comprehensive framework for label noise detection and correction in regression tasks, supporting multiple regression backends and deep learning models.

## Overview

BVRC implements a 3-stage algorithm for identifying and correcting noisy labels in regression datasets:

1. **Stage 1: Noise Identification** - Uses cross-validation to identify potentially noisy samples
2. **Stage 2: Label Prediction** - Predicts corrected labels using clean samples
3. **Stage 3: Correction** - Applies adaptive confidence intervals for final correction

The framework supports three different regression backends:
- **BVRC-X**: XGBoost (fast, recommended for large datasets)
- **BVRC-T**: TabPFN (accurate, recommended for small datasets <10K samples)
- **BVRC-D**: Diffusion Model (requires pre-trained model)

## Project Structure

```
BVRC_code/
├── BVRC_main.py              # Main correction script with unified interface
├── regression_models/        # Regression model implementations
│   ├── __init__.py          # Factory function for model instantiation
│   ├── BVRC_X.py            # XGBoost regressor wrapper
│   ├── BVRC_T.py            # TabPFN regressor wrapper
│   └── BVRC_D.py            # Diffusion regressor wrapper
├── script/
│   └── train.py             # Training script for evaluating corrected labels
├── models/                   # Directory for pre-trained models
│   ├── (place your pretrained checkpoints here)
│   └── (e.g., resnet50_baseline.pth, vit_model/)
├── datasets/                 # Directory for datasets
│   └── (place your datasets here)
├── results/                  # Output directory for correction results
├── requirements.txt          # Python dependencies
├── INSTALL.md                # Installation guide
└── README.md                 # This file
```

## Datasets

BVRC has been tested on multiple regression datasets across different domains. Below are the supported datasets with download links.

### Age Estimation Datasets

| Dataset | Domain | Samples | Label | Download Link |
|---------|--------|---------|-------|---------------|
| **UTKFace** | Face Age | ~23K | Age (0-116) | [Download](https://susanqq.github.io/UTKFace/) |
| **MORPH** | Face Age | ~55K | Age (16-77) | [Download](https://uncw.edu/research/innovation/commercialization/technology-portfolio/morph.html) |
| **AgeDB** | Face Age | ~16K | Age (1-101) | [Download](https://ibug.doc.ic.ac.uk/resources/agedb/) |
| **CLAP2016** | Face Age | ~7.5K | Age (1-95) | [Download](https://chalearnlap.cvc.uab.cat/dataset/26/description/) |

### Other Regression Datasets

| Dataset | Domain | Samples | Label | Download Link |
|---------|--------|---------|-------|---------------|
| **RSNA Bone Age** | Medical | ~12K | Bone Age (months) | [Kaggle](https://www.kaggle.com/datasets/kmader/rsna-bone-age) |
| **ForestDK** | Ecology | ~40K | Tree Cover (%) | [Download](https://sid.erda.dk/share_redirect/f1Hmpeh6O2) |
| **MPIIFaceGaze** | Gaze Estimation | ~45K | Gaze Angle (degrees) | [Download](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2Fdarus-3240) |

### Dataset Preparation

1. **Download** your chosen dataset from the links above
2. **Organize** the dataset in the following structure:

```
datasets/
└── {dataset_name}/
    ├── images/              # All image files
    │   ├── image001.jpg
    │   ├── image002.jpg
    │   └── ...
    ├── train.csv           # Training annotations
    ├── val.csv             # Validation annotations (optional)
    └── test.csv            # Test annotations
```

3. **Prepare CSV files** with the following format:

```csv
filename,age
image001.jpg,25
image002.jpg,32
image003.jpg,18
...
```

**CSV Requirements:**
- `filename`: Image filename (must match files in `images/` directory)
- `age`: Label value (can be renamed via `--label_col` parameter)
- Additional columns are allowed but will be ignored

### Example: UTKFace Dataset

After downloading UTKFace:

```bash
# 1. Extract images
unzip UTKFace.zip -d datasets/UTKFace/images/

# 2. Create CSV files (example using Python)
python -c "
import os
import pandas as pd

image_dir = 'datasets/UTKFace/images/'
data = []

for fname in os.listdir(image_dir):
    if fname.endswith('.jpg'):
        age = int(fname.split('_')[0])  # UTKFace format: age_gender_race_date.jpg
        data.append({'filename': fname, 'age': age})

df = pd.DataFrame(data)
df.sample(frac=0.8, random_state=42).to_csv('datasets/UTKFace/train.csv', index=False)
df.drop(df.sample(frac=0.8, random_state=42).index).to_csv('datasets/UTKFace/test.csv', index=False)
"

# 3. Run BVRC
python BVRC_main.py \
    --regressor X \
    --annotations datasets/UTKFace/train.csv \
    --image_dir datasets/UTKFace/images \
    --checkpoint models/resnet50_baseline.pth \
    --output_dir results/
```

### Custom Datasets

To use your own dataset:

1. Organize images in a single directory
2. Create a CSV file with at least `filename` and label columns
3. Adjust `--filename_col` and `--label_col` parameters if using different column names

**Example with custom columns:**

```bash
python BVRC_main.py \
    --regressor X \
    --annotations my_dataset.csv \
    --image_dir my_images/ \
    --filename_col "image_path" \
    --label_col "target_value" \
    --checkpoint models/resnet50_baseline.pth \
    --output_dir results/
```

## Installation

### Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended)
- 8GB+ RAM (16GB+ recommended for Diffusion model)

### Setup

1. Clone or download this repository

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. For GPU acceleration with XGBoost, ensure your CUDA toolkit is installed and matches your PyTorch version.

## Usage

### 1. Label Noise Correction

The main script `BVRC_main.py` provides a unified interface for all three regression backends.

#### Basic Usage with XGBoost (BVRC-X)

```bash
python BVRC_main.py \
    --regressor X \
    --annotations train.csv \
    --image_dir /path/to/images \
    --checkpoint models/resnet50_baseline.pth \
    --output_dir ./results \
    --noise_threshold 2.0 \
    --std_threshold 2.0
```

#### Using TabPFN (BVRC-T)

Recommended for small datasets (<10K samples). Includes PCA for dimensionality reduction.

```bash
python BVRC_main.py \
    --regressor T \
    --use_pca \
    --pca_components 400 \
    --annotations train.csv \
    --image_dir /path/to/images \
    --checkpoint models/resnet50_baseline.pth \
    --output_dir ./results
```

#### Using Diffusion Model (BVRC-D)

Requires pre-trained diffusion model and CLIP feature encoder.

```bash
python BVRC_main.py \
    --regressor D \
    --clip_path /path/to/clip-vit-base \
    --diffusion_model models/diffusion_pretrained.pt \
    --annotations train.csv \
    --image_dir /path/to/images \
    --checkpoint models/resnet50_baseline.pth \
    --output_dir ./results \
    --finetune_epochs 5
```

#### Key Parameters

- `--regressor`: Choose regression backend (`X`=XGBoost, `T`=TabPFN, `D`=Diffusion)
- `--annotations`: Path to CSV file with columns: `filename`, `age` (or custom label)
- `--image_dir`: Directory containing images
- `--checkpoint`: Pre-trained feature extractor checkpoint
- `--model`: Feature extractor architecture (default: `resnet50`)
- `--noise_threshold`: Sigma threshold for noise identification (default: 2.0)
- `--std_threshold`: Standard deviation threshold for correction (default: 2.0)
- `--n_folds`: Number of cross-validation folds (default: 5)

### 2. Model Training and Evaluation

Use `script/train.py` to train models on corrected datasets and automatically evaluate on test sets. The script supports training with validation monitoring and final test evaluation.

#### Supported Models

- **ResNet50** - Recommended for most tasks
- **VGG16** - Good for transfer learning
- **Vision Transformer (ViT)** - Best accuracy, requires more resources

#### Train with ResNet50

```bash
python script/train.py \
    --model resnet50 \
    --train_annotations results/corrected_dataset.csv \
    --train_image_dir /path/to/images \
    --val_annotations val.csv \
    --val_image_dir /path/to/images \
    --test_annotations test.csv \
    --test_image_dir /path/to/images \
    --output_dir ./runs \
    --experiment_tag "BVRC_corrected" \
    --epochs 100 \
    --batch_size 128 \
    --lr 1e-4 \
    --augmentation medium \
    --early_stopping \
    --use_amp
```

#### Train with VGG16

VGG16 works well with transfer learning. Use smaller learning rate when fine-tuning.

```bash
python script/train.py \
    --model vgg16 \
    --train_annotations results/corrected_dataset.csv \
    --train_image_dir /path/to/images \
    --val_annotations val.csv \
    --val_image_dir /path/to/images \
    --test_annotations test.csv \
    --test_image_dir /path/to/images \
    --output_dir ./runs \
    --experiment_tag "VGG16_corrected" \
    --epochs 100 \
    --batch_size 128 \
    --lr 1e-4 \
    --augmentation medium \
    --early_stopping \
    --use_amp
```

#### Train with ViT

Vision Transformer requires pre-trained model path and typically uses lower learning rate.

```bash
python script/train.py \
    --model vit \
    --vit_model_path models/vit-base-patch16-224-in21k \
    --train_annotations results/corrected_dataset.csv \
    --train_image_dir /path/to/images \
    --val_annotations val.csv \
    --val_image_dir /path/to/images \
    --test_annotations test.csv \
    --test_image_dir /path/to/images \
    --output_dir ./runs \
    --experiment_tag "ViT_corrected" \
    --epochs 100 \
    --batch_size 64 \
    --lr 2e-5 \
    --early_stopping \
    --use_amp
```

**Note**: ViT uses its own image preprocessing via `ViTImageProcessor`, so `--augmentation` parameter is ignored for ViT models.

#### Key Parameters

**Model Configuration:**
- `--model`: Model architecture (`resnet50`, `vgg16`, `vit`)
- `--vit_model_path`: Path to pre-trained ViT model (required for `vit`)
- `--freeze_backbone`: Freeze backbone weights for transfer learning

**Training Hyperparameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (ResNet/VGG: 1e-4, ViT: 2e-5)
- `--weight_decay`: Weight decay for regularization (default: 1e-4)
- `--gradient_clip`: Gradient clipping threshold (default: 1.0)

**Data Augmentation:**
- `--augmentation`: Augmentation strength for CNN models (`none`, `light`, `medium`, `strong`)
- `--image_size`: Input image size (default: 256)

**Optimization:**
- `--optimizer`: Optimizer type (`adam`, `adamw`, `sgd`)
- `--scheduler`: Learning rate scheduler (`plateau`, `cosine`, `none`)
- `--scheduler_patience`: Patience for ReduceLROnPlateau (default: 10)

**Training Control:**
- `--early_stopping`: Enable early stopping
- `--early_stopping_patience`: Patience for early stopping (default: 15)
- `--use_amp`: Use automatic mixed precision training (recommended)

## Output

### Correction Results

After running `BVRC_main.py`, you'll find in the output directory:

```
output_dir/
└── BVRC_{regressor}_{threshold}/
    ├── {threshold}_corrected_samples.csv      # Only corrected samples
    ├── {threshold}_corrected_dataset.csv      # Full dataset with corrections
    └── {threshold}_stats.txt                  # Correction statistics
```

**Correction Statistics Include:**
- Number of noisy samples identified
- Number of labels corrected
- Noise ratio
- Mean and maximum correction magnitude
- Execution time

### Training Results

After running `script/train.py`, you'll find:

```
output_dir/
└── {model}_{experiment_tag}_{timestamp}/
    ├── best_checkpoint.pth                    # Best model weights
    ├── checkpoint_epoch_{N}.pth              # Periodic checkpoints
    ├── training_history.csv                   # Training metrics per epoch
    ├── training.log                           # Detailed training log
    └── test_results.txt                       # Final test evaluation
```

## Examples

### Complete Workflow Example

```bash
# Step 1: Extract features and correct labels using XGBoost
python BVRC_main.py \
    --regressor X \
    --annotations data/train_noisy.csv \
    --image_dir data/images \
    --checkpoint models/resnet50_baseline.pth \
    --output_dir ./results \
    --noise_threshold 2.0

# Step 2: Train model on corrected dataset
python script/train.py \
    --model resnet50 \
    --train_annotations results/BVRC_XGBoost_2.0sigma_2.0std/2.0sigma_2.0std_corrected_dataset.csv \
    --train_image_dir data/images \
    --val_annotations data/val.csv \
    --val_image_dir data/images \
    --test_annotations data/test.csv \
    --test_image_dir data/images \
    --output_dir ./runs \
    --experiment_tag "corrected_baseline" \
    --epochs 100 \
    --early_stopping \
    --use_amp

# Step 3: Compare with baseline trained on noisy labels
python script/train.py \
    --model resnet50 \
    --train_annotations data/train_noisy.csv \
    --train_image_dir data/images \
    --val_annotations data/val.csv \
    --val_image_dir data/images \
    --test_annotations data/test.csv \
    --test_image_dir data/images \
    --output_dir ./runs \
    --experiment_tag "noisy_baseline" \
    --epochs 100
```

## Regression Model Details

### BVRC-X (XGBoost)

- **Pros**: Fast, scalable, works well for large datasets
- **Cons**: May require GPU for very large datasets
- **Best for**: General-purpose correction, large-scale experiments

### BVRC-T (TabPFN)

- **Pros**: State-of-the-art accuracy on small datasets, no hyperparameter tuning
- **Cons**: Limited to <10K samples, requires PCA for high-dimensional features
- **Best for**: Small to medium datasets where accuracy is critical

### BVRC-D (Diffusion Model)

- **Pros**: Can capture complex data distributions, flexible
- **Cons**: Requires pre-trained model, computationally expensive, needs careful tuning
- **Best for**: Research experiments with available pre-trained diffusion models

## Tips and Best Practices

1. **Feature Extractor**: Use a feature extractor trained on your target domain for best results
2. **Threshold Selection**: Start with default thresholds (2.0, 2.0) and adjust based on validation
3. **Cross-validation Folds**: 5 folds is a good default; increase for smaller datasets
4. **TabPFN + PCA**: Always use PCA when feature dimension > 500
5. **Data Augmentation**: Use `medium` augmentation for most cases; `strong` for small datasets
6. **Early Stopping**: Enable for efficiency; patience of 15 epochs works well

## Troubleshooting

### Out of Memory

- Reduce `--batch_size`
- Use `--use_pca` with TabPFN
- For Diffusion model, reduce `--finetune_epochs`

### Poor Correction Quality

- Adjust `--noise_threshold` (try 1.5-3.0 range)
- Increase `--n_folds` for more robust predictions
- Ensure feature extractor is well-trained

### Slow Performance

- Use BVRC-X instead of BVRC-T or BVRC-D
- Enable GPU acceleration
- Reduce number of cross-validation folds

## Citation



## License


## Contact

