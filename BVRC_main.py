"""
BVRC: Bias-Variance Risk-Controlled Correction for Label Noise
Main script for label noise detection and correction in regression tasks.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from scipy.stats import gaussian_kde


# ============================================================================
#                           Feature Extractor
# ============================================================================

def get_feature_extractor(model_name: str, custom_model=None):
    """
    Build a headless feature extraction model.
    
    Args:
        model_name: 'resnet50' or 'custom'
        custom_model: Custom model instance (when model_name='custom')
    
    Returns:
        A pretrained model with the classification head removed.
    """
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()
        return model
    elif model_name == 'custom' and custom_model is not None:
        return custom_model
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# ============================================================================
#                           Dataset
# ============================================================================

class FeatureExtractionDataset(Dataset):
    """Dataset for feature extraction with standard image transforms."""
    
    def __init__(self, annotations_df, img_dir, filename_col, label_col):
        self.annotations_df = annotations_df
        self.img_dir = img_dir
        self.filename_col = filename_col
        self.label_col = label_col
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        row = self.annotations_df.iloc[idx]
        filename = row[self.filename_col]
        label = torch.tensor(row[self.label_col], dtype=torch.float32)
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label, filename


# ============================================================================
#                           Feature Extraction
# ============================================================================

def extract_features(model, dataloader, device):
    """Extract deep features using pretrained model."""
    model.eval()
    all_features, all_labels, all_filenames = [], [], []
    
    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_filenames.extend(filenames)
            
    return np.vstack(all_features), np.array(all_labels), all_filenames


# ============================================================================
#                           K-Fold Utilities
# ============================================================================

def stratified_kfold_indices(labels, n_splits=5, n_bins=10, random_state=42):
    """
    Create stratified K-fold indices for regression tasks.
    Bins continuous labels and distributes samples evenly across folds.
    """
    np.random.seed(random_state)
    n_samples = len(labels)
    
    bins = np.percentile(labels, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    digitized = np.digitize(labels, bins)
    
    fold_ids = np.zeros(n_samples, dtype=int)
    
    for bin_idx in np.unique(digitized):
        bin_mask = digitized == bin_idx
        bin_indices = np.where(bin_mask)[0]
        np.random.shuffle(bin_indices)
        for i, idx in enumerate(bin_indices):
            fold_ids[idx] = (i % n_splits) + 1
    
    return fold_ids


# ============================================================================
#                           BVRC Core Algorithm
# ============================================================================

def compute_kde_statistics(samples):
    """Compute mode, mean, and std using KDE."""
    if len(samples) < 2 or np.std(samples) < 1e-6:
        val = np.mean(samples)
        return val, val, 0.0
    
    try:
        kde = gaussian_kde(samples)
        xi = np.linspace(samples.min() - 1, samples.max() + 1, 1000)
        f = kde(xi)
        mode = xi[np.argmax(f)]
        dx = xi[1] - xi[0]
        mean = np.trapz(xi * f, dx=dx)
        second_moment = np.trapz(xi**2 * f, dx=dx)
        std = np.sqrt(max(0, second_moment - mean**2))
        return mode, mean, std
    except:
        return np.median(samples), np.mean(samples), np.std(samples)


def BVRC(features, labels, regressor, threshold=3.0, std_threshold=2.0, 
         n_folds=5, random_seed=42):
    """
    BVRC: Bias-Variance Risk-Controlled Correction Algorithm.
    
    Args:
        features: Feature array [n_samples, n_features]
        labels: Label array [n_samples]
        regressor: Regressor instance with fit() and predict() methods
        threshold: Sigma threshold for noise identification (Stage 1)
        std_threshold: Std threshold for correction (Stage 2)
        n_folds: Number of cross-validation folds
        random_seed: Random seed for reproducibility
    
    Returns:
        corrected_labels: Corrected label array
        stats: Dictionary of statistics
    """
    n_samples = len(labels)
    t_start = time.time()

    # Stage 1: Noise Identification
    print("=" * 60)
    print("Stage 1: Noise Identification")
    print("=" * 60)
    
    fold_ids = stratified_kfold_indices(labels, n_splits=n_folds, random_state=random_seed)
    predictions_stage1 = np.zeros((n_samples, n_folds))
    
    for fold in tqdm(range(1, n_folds + 1), desc="Cross-validation"):
        fold_mask = (fold_ids == fold)
        X_fold = features[fold_mask]
        y_fold = labels[fold_mask]
        
        reg = regressor.clone() if hasattr(regressor, 'clone') else regressor
        reg.fit(X_fold, y_fold)
        predictions_stage1[:, fold-1] = reg.predict(features)
    
    # Compute KDE-based predictions
    kde_predictions = np.array([compute_kde_statistics(predictions_stage1[i])[0] 
                                for i in range(n_samples)])
    
    # Apply sigma-rule for noise detection
    residuals = kde_predictions - labels
    mean_res, std_res = np.mean(residuals), np.std(residuals)
    lower_bound = mean_res - threshold * std_res
    upper_bound = mean_res + threshold * std_res
    
    noise_mask = (residuals < lower_bound) | (residuals > upper_bound)
    clean_mask = ~noise_mask
    
    noise_count = np.sum(noise_mask)
    print(f"Identified {np.sum(clean_mask)} clean samples, {noise_count} noisy samples")
    print(f"Noise ratio: {noise_count / n_samples * 100:.2f}%")

    if noise_count == 0:
        print("No noisy samples found.")
        return labels.copy(), {
            'noise_count': 0, 'corrected_count': 0, 'noise_ratio': 0.0,
            'execution_time': time.time() - t_start
        }

    # Stage 2: Prediction using clean samples
    print("\n" + "=" * 60)
    print("Stage 2: Label Prediction for Noisy Samples")
    print("=" * 60)
    
    X_clean, y_clean = features[clean_mask], labels[clean_mask]
    X_noise, y_noise = features[noise_mask], labels[noise_mask]
    
    fold_ids_clean = stratified_kfold_indices(y_clean, n_splits=n_folds, random_state=random_seed)
    predictions_stage2 = np.zeros((noise_count, n_folds))
    
    for fold in tqdm(range(1, n_folds + 1), desc="Prediction"):
        fold_mask = (fold_ids_clean == fold)
        X_fold = X_clean[fold_mask]
        y_fold = y_clean[fold_mask]
        
        reg = regressor.clone() if hasattr(regressor, 'clone') else regressor
        reg.fit(X_fold, y_fold)
        predictions_stage2[:, fold-1] = reg.predict(X_noise)

    # Stage 3: Confidence interval computation and correction
    print("\n" + "=" * 60)
    print("Stage 3: Label Correction")
    print("=" * 60)
    
    f_x = np.zeros(noise_count)
    EY = np.zeros(noise_count)
    std_devs = np.zeros(noise_count)
    
    for i in range(noise_count):
        mode, mean, std = compute_kde_statistics(predictions_stage2[i])
        f_x[i], EY[i], std_devs[i] = mode, mean, std

    # Compute adaptive threshold T
    TT = 2 * (f_x - y_noise) * (f_x - EY)
    T1 = np.mean(np.sqrt(np.maximum(TT, 0)))
    T2 = np.sqrt(np.maximum(TT, 0))
    T = (T1 + T2) / 2
    
    Interval = np.column_stack([f_x - T, f_x + T])
    
    # Correction conditions
    condition1 = (y_noise < Interval[:, 0]) | (y_noise > Interval[:, 1])
    condition2 = np.abs(y_noise - f_x) > std_threshold * std_devs
    correct_mask = condition1 & condition2
    
    corrected_count = np.sum(correct_mask)
    print(f"Samples outside confidence interval: {np.sum(condition1)}")
    print(f"Samples exceeding {std_threshold}*std: {np.sum(condition2)}")
    print(f"Samples to correct: {corrected_count}")
    
    # Apply correction
    corrected_labels = labels.copy()
    noise_indices = np.where(noise_mask)[0]
    corrected_labels[noise_indices[correct_mask]] = f_x[correct_mask]

    elapsed_time = time.time() - t_start
    
    stats = {
        'noise_count': int(noise_count),
        'corrected_count': int(corrected_count),
        'noise_ratio': float(noise_count / n_samples),
        'execution_time': float(elapsed_time),
        'threshold': threshold,
        'std_threshold': std_threshold
    }
    
    return corrected_labels, stats


# ============================================================================
#                           Main Function
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='BVRC: Label Noise Correction for Regression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using XGBoost (default)
  python BVRC_main.py --regressor X --annotations train.csv --image_dir images/ ...
  
  # Using TabPFN (for small datasets)
  python BVRC_main.py --regressor T --use_pca --pca_components 400 ...
  
  # Using Diffusion (requires pre-trained model)
  python BVRC_main.py --regressor D --clip_path /path/to/clip --diffusion_model /path/to/model.pt ...
        """
    )
    
    # Required arguments
    parser.add_argument('--annotations', type=str, required=True,
                        help='Path to CSV file with annotations')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to image directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint for feature extraction')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    
    # Regressor selection
    parser.add_argument('--regressor', type=str, default='X', choices=['X', 'T', 'D'],
                        help='Regressor type: X=XGBoost, T=TabPFN, D=Diffusion')
    
    # Feature extraction
    parser.add_argument('--model', type=str, default='resnet50',
                        help='Model architecture for feature extraction')
    parser.add_argument('--filename_col', type=str, default='filename',
                        help='Column name for filenames in CSV')
    parser.add_argument('--label_col', type=str, default='age',
                        help='Column name for labels in CSV')
    
    # BVRC parameters
    parser.add_argument('--noise_threshold', type=float, default=3.0,
                        help='Sigma threshold for noise identification')
    parser.add_argument('--std_threshold', type=float, default=2.0,
                        help='Std threshold for label correction')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # XGBoost specific
    parser.add_argument('--xgb_gpu_id', type=int, default=0,
                        help='GPU ID for XGBoost')
    
    # TabPFN specific
    parser.add_argument('--use_pca', action='store_true',
                        help='Enable PCA for TabPFN')
    parser.add_argument('--pca_components', type=int, default=400,
                        help='Number of PCA components')
    parser.add_argument('--tabpfn_device', type=str, default='cuda:0',
                        help='Device for TabPFN')
    
    # Diffusion specific
    parser.add_argument('--clip_path', type=str, default=None,
                        help='Path to CLIP model for Diffusion')
    parser.add_argument('--diffusion_model', type=str, default=None,
                        help='Path to pre-trained diffusion model')
    parser.add_argument('--finetune_epochs', type=int, default=5,
                        help='Finetune epochs for Diffusion')
    
    # System
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for PyTorch')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup output directory
    regressor_names = {'X': 'XGBoost', 'T': 'TabPFN', 'D': 'Diffusion'}
    output_subdir = f"BVRC_{regressor_names[args.regressor]}_{args.noise_threshold}sigma_{args.std_threshold}std"
    output_dir = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Regressor: {regressor_names[args.regressor]}")

    # Load feature extraction model
    print("\n" + "=" * 60)
    print(f"Loading {args.model} feature extractor")
    print("=" * 60)
    
    model = get_feature_extractor(args.model)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    print(f"Model loaded from {args.checkpoint}")

    # Prepare dataloader
    print("\n" + "=" * 60)
    print("Preparing dataloader")
    print("=" * 60)
    
    df = pd.read_csv(args.annotations)
    dataset = FeatureExtractionDataset(df, args.image_dir, args.filename_col, args.label_col)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Loaded {len(df)} samples")
    
    # Extract features
    print("\n" + "=" * 60)
    print("Extracting features")
    print("=" * 60)
    
    features, labels, filenames = extract_features(model, dataloader, device)
    print(f"Extracted features: {features.shape}")
    
    # Release feature extractor memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize regressor
    print("\n" + "=" * 60)
    print(f"Initializing {regressor_names[args.regressor]} regressor")
    print("=" * 60)
    
    from regression_models import get_regressor
    
    if args.regressor == 'X':
        regressor = get_regressor('X', gpu_id=args.xgb_gpu_id, random_state=args.random_seed)
    elif args.regressor == 'T':
        regressor = get_regressor('T', device=args.tabpfn_device, use_pca=args.use_pca,
                                  pca_components=args.pca_components, random_state=args.random_seed)
    elif args.regressor == 'D':
        if not args.clip_path or not args.diffusion_model:
            raise ValueError("--clip_path and --diffusion_model required for Diffusion regressor")
        regressor = get_regressor('D', clip_path=args.clip_path, model_path=args.diffusion_model,
                                  device=args.device, finetune_epochs=args.finetune_epochs,
                                  y_mean=labels.mean(), y_std=labels.std())
        # For diffusion, we need to load images
        print("Loading images for Diffusion regressor...")
        from PIL import Image
        from regression_models.BVRC_D import ViTWrapper
        vit = ViTWrapper(args.clip_path, device=args.device)
        images = []
        for fn in tqdm(filenames, desc="Loading images"):
            img = Image.open(os.path.join(args.image_dir, fn)).convert('RGB')
            images.append(vit.transform(img))
        regressor.set_images(torch.stack(images))
        regressor.set_normalization_stats(labels.mean(), labels.std())
        del vit
        torch.cuda.empty_cache()
    
    print(f"Regressor initialized: {regressor.get_name()}")

    # Run BVRC algorithm
    print("\n" + "=" * 60)
    print("Running BVRC algorithm")
    print("=" * 60)
    
    corrected_labels, stats = BVRC(
        features, labels, regressor,
        threshold=args.noise_threshold,
        std_threshold=args.std_threshold,
        n_folds=args.n_folds,
        random_seed=args.random_seed
    )

    # Save results
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)
    
    results_df = pd.DataFrame({
        'filename': filenames,
        'original_label': labels,
        'corrected_label': corrected_labels,
        'difference': corrected_labels - labels
    })
    
    corrected_samples = results_df[results_df['difference'] != 0]
    num_corrected = len(corrected_samples)
    
    print(f"\nSummary:")
    print(f"  Total samples: {len(results_df)}")
    print(f"  Noisy samples: {stats['noise_count']} ({stats['noise_ratio']*100:.2f}%)")
    print(f"  Corrected: {num_corrected} ({num_corrected/len(results_df)*100:.2f}%)")
    
    tag = f"{args.noise_threshold}sigma_{args.std_threshold}std"
    
    # Save files
    corrected_samples.to_csv(os.path.join(output_dir, f"{tag}_corrected_samples.csv"), index=False)
    full_df = results_df[['filename', 'corrected_label']].copy()
    full_df.rename(columns={'corrected_label': args.label_col}, inplace=True)
    full_df.to_csv(os.path.join(output_dir, f"{tag}_corrected_dataset.csv"), index=False)
    
    # Save statistics
    with open(os.path.join(output_dir, f"{tag}_stats.txt"), 'w') as f:
        f.write(f"BVRC Label Correction Statistics ({regressor_names[args.regressor]})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Regressor: {regressor_names[args.regressor]}\n")
        f.write(f"Noise threshold: {args.noise_threshold} sigma\n")
        f.write(f"Std threshold: {args.std_threshold}\n")
        f.write(f"N folds: {args.n_folds}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Total samples: {len(results_df)}\n")
        f.write(f"  Noisy samples: {stats['noise_count']} ({stats['noise_ratio']*100:.2f}%)\n")
        f.write(f"  Corrected: {num_corrected} ({num_corrected/len(results_df)*100:.2f}%)\n")
        if num_corrected > 0:
            f.write(f"  Mean correction: {np.abs(corrected_samples['difference']).mean():.2f}\n")
            f.write(f"  Max correction: {np.abs(corrected_samples['difference']).max():.2f}\n")
        f.write(f"  Execution time: {stats['execution_time']:.2f}s\n")
    
    print(f"\nResults saved to: {output_dir}")
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
