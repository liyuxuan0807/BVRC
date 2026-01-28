"""
Training script for regression models with label noise correction evaluation.
Supports ResNet50, VGG16, and ViT backbones with comprehensive training utilities.
"""

import os
import time
import random
import argparse
import logging
import multiprocessing
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from transformers import ViTForImageClassification, ViTImageProcessor


# ============================================================================
#                           Utilities
# ============================================================================

def setup_logging(log_path):
    """Configure logging to file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_num_workers(num_workers='auto'):
    """Get optimal number of data loader workers."""
    if num_workers == 'auto':
        return min(8, multiprocessing.cpu_count() // 2)
    return int(num_workers)


# ============================================================================
#                           Models
# ============================================================================

def get_vgg16_for_regression(freeze_features=True, output_dim=1):
    """
    Build VGG16 model for regression task.
    
    Args:
        freeze_features: Whether to freeze feature extraction layers
        output_dim: Output dimension (default: 1 for single value regression)
    
    Returns:
        VGG16 model configured for regression
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    
    num_input_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features=num_input_features, out_features=output_dim)
    
    logging.info("VGG16 model loaded for regression")
    logging.info(f"Classifier head: {model.classifier}")
    
    return model


def get_resnet50_for_regression(freeze_features=True, output_dim=1):
    """
    Build ResNet50 model for regression task.
    
    Args:
        freeze_features: Whether to freeze feature extraction layers
        output_dim: Output dimension (default: 1 for single value regression)
    
    Returns:
        ResNet50 model configured for regression
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
    
    num_input_features = model.fc.in_features
    model.fc = nn.Linear(in_features=num_input_features, out_features=output_dim)
    
    logging.info("ResNet50 model loaded for regression")
    logging.info(f"FC layer: {model.fc}")
    
    return model


def get_vit_for_regression(vit_model_path, freeze_backbone=True, output_dim=1):
    """
    Build Vision Transformer (ViT) model for regression task.
    
    Args:
        vit_model_path: Path to pretrained ViT model
        freeze_backbone: Whether to freeze backbone layers
        output_dim: Output dimension (default: 1 for single value regression)
    
    Returns:
        ViT model configured for regression
    """
    model = ViTForImageClassification.from_pretrained(
        vit_model_path,
        num_labels=output_dim,
        ignore_mismatched_sizes=True
    )
    
    if freeze_backbone:
        for param in model.vit.parameters():
            param.requires_grad = False
    
    logging.info("ViT model loaded for regression")
    logging.info(f"Classifier head: {model.classifier}")
    
    return model


def get_model(model_name, freeze_backbone=False, vit_model_path=None, output_dim=1):
    """
    Build regression model based on model name.
    
    Args:
        model_name: 'resnet50', 'vgg16', or 'vit'
        freeze_backbone: Whether to freeze backbone weights
        vit_model_path: Path to ViT model (required if model_name='vit')
        output_dim: Output dimension
    
    Returns:
        Model configured for regression
    """
    if model_name == 'resnet50':
        return get_resnet50_for_regression(freeze_backbone, output_dim)
    elif model_name == 'vgg16':
        return get_vgg16_for_regression(freeze_backbone, output_dim)
    elif model_name == 'vit':
        if vit_model_path is None:
            raise ValueError("vit_model_path is required for ViT model")
        return get_vit_for_regression(vit_model_path, freeze_backbone, output_dim)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from 'resnet50', 'vgg16', 'vit'")


# ============================================================================
#                           Early Stopping
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=15, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric):
        if self.best_score is None:
            self.best_score = metric
            return False
        
        improved = (metric < self.best_score - self.min_delta) if self.mode == 'min' \
                   else (metric > self.best_score + self.min_delta)
        
        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


# ============================================================================
#                           Dataset
# ============================================================================

class RegressionDataset(Dataset):
    """Dataset for regression tasks with image inputs."""
    
    def __init__(self, annotations_df, img_dir, filename_col, label_col, image_size=256):
        self.df = annotations_df
        self.img_dir = img_dir
        self.filename_col = filename_col
        self.label_col = label_col
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row[self.filename_col])
        label = torch.tensor(row[self.label_col], dtype=torch.float32)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.warning(f"Failed to load image {img_path}: {e}. Using placeholder.")
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        
        return image, label.unsqueeze(-1)


def get_transforms(image_size=256, augmentation='medium'):
    """
    Get image transforms for training and validation.
    
    Args:
        image_size: Target image size
        augmentation: 'none', 'light', 'medium', 'strong'
    
    Returns:
        train_transform, val_transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Training transform with augmentation
    if augmentation == 'none':
        train_transform = val_transform
    elif augmentation == 'light':
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
    elif augmentation == 'medium':
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            normalize
        ])
    elif augmentation == 'strong':
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1)
        ])
    else:
        train_transform = val_transform
    
    return train_transform, val_transform


def get_dataloaders(args):
    """Create data loaders for training and validation."""
    num_workers = get_num_workers(args.num_workers)
    logging.info(f"Using {num_workers} data loader workers")
    
    collate_fn_train = None
    collate_fn_val = None
    
    # ViT requires special collate function with ViTImageProcessor
    if args.model == 'vit':
        logging.info("Configuring ViT-specific collate function...")
        image_processor = ViTImageProcessor.from_pretrained(args.vit_model_path)
        
        def vit_collate_fn(batch):
            images, labels = zip(*batch)
            inputs = image_processor(list(images), return_tensors="pt")
            return inputs['pixel_values'], torch.stack(labels)
        
        collate_fn_train = vit_collate_fn
        collate_fn_val = vit_collate_fn
    else:
        # CNN models (ResNet50, VGG16)
        logging.info(f"Configuring CNN collate function for {args.model}...")
        train_transform, val_transform = get_transforms(args.image_size, args.augmentation)
        
        if args.augmentation != 'none':
            logging.info(f"Training augmentation: {args.augmentation}")
        
        def train_collate_fn(batch):
            images, labels = zip(*batch)
            processed = [train_transform(img) for img in images]
            return torch.stack(processed), torch.stack(labels)
        
        def val_collate_fn(batch):
            images, labels = zip(*batch)
            processed = [val_transform(img) for img in images]
            return torch.stack(processed), torch.stack(labels)
        
        collate_fn_train = train_collate_fn
        collate_fn_val = val_collate_fn
    
    # Load data
    train_df = pd.read_csv(args.train_annotations)
    val_df = pd.read_csv(args.val_annotations)
    logging.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    train_dataset = RegressionDataset(
        train_df, args.train_image_dir, args.filename_col, args.label_col, args.image_size
    )
    val_dataset = RegressionDataset(
        val_df, args.val_image_dir, args.filename_col, args.label_col, args.image_size
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_train
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_val
    )
    
    return train_loader, val_loader, collate_fn_val


# ============================================================================
#                           Training
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, model_name,
                    use_amp=False, gradient_clip=None, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                if model_name == 'vit':
                    outputs = model(pixel_values=inputs).logits
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            if gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            if model_name == 'vit':
                outputs = model(pixel_values=inputs).logits
            else:
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, device, model_name):
    """Validate model and compute metrics."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs = inputs.to(device)
            
            if model_name == 'vit':
                outputs = model(pixel_values=inputs).logits
            else:
                outputs = model(inputs)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    return {
        "mse": mean_squared_error(all_labels, all_preds),
        "mae": mean_absolute_error(all_labels, all_preds),
        "r2": r2_score(all_labels, all_preds)
    }


# ============================================================================
#                           Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train regression model for label noise evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with ResNet50
  python train.py --model resnet50 --train_annotations train.csv ...
  
  # Train with VGG16
  python train.py --model vgg16 --train_annotations train.csv ...
  
  # Train with ViT (requires vit_model_path)
  python train.py --model vit --vit_model_path /path/to/vit --train_annotations train.csv ...
        """
    )
    
    # Data paths
    parser.add_argument('--train_annotations', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--train_image_dir', type=str, required=True,
                        help='Path to training images directory')
    parser.add_argument('--val_annotations', type=str, required=True,
                        help='Path to validation CSV file')
    parser.add_argument('--val_image_dir', type=str, required=True,
                        help='Path to validation images directory')
    parser.add_argument('--test_annotations', type=str, default=None,
                        help='Path to test CSV file (optional)')
    parser.add_argument('--test_image_dir', type=str, default=None,
                        help='Path to test images directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints and logs')
    
    # Data columns
    parser.add_argument('--filename_col', type=str, default='filename',
                        help='Column name for filenames')
    parser.add_argument('--label_col', type=str, default='age',
                        help='Column name for labels')
    
    # Model
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'vgg16', 'vit'],
                        help='Model architecture: resnet50, vgg16, or vit')
    parser.add_argument('--vit_model_path', type=str, default=None,
                        help='Path to pretrained ViT model (required for vit)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone weights')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    
    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=10,
                        help='Patience for ReduceLROnPlateau')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor for learning rate reduction')
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001,
                        help='Minimum improvement for early stopping')
    
    # Data augmentation
    parser.add_argument('--augmentation', type=str, default='medium',
                        choices=['none', 'light', 'medium', 'strong'],
                        help='Data augmentation strength')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size')
    
    # Training settings
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for training')
    parser.add_argument('--num_workers', type=str, default='auto',
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Logging
    parser.add_argument('--experiment_tag', type=str, default='',
                        help='Experiment tag for output directory')
    parser.add_argument('--save_every', type=int, default=20,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate ViT model path
    if args.model == 'vit' and args.vit_model_path is None:
        raise ValueError("--vit_model_path is required when using ViT model")
    
    # Setup output directory
    tag = f"_{args.experiment_tag}" if args.experiment_tag else ""
    run_dir = os.path.join(
        args.output_dir,
        f"{args.model}{tag}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(os.path.join(run_dir, 'training.log'))
    
    # Set seed and device
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    logging.info("=" * 70)
    logging.info("Regression Training Script")
    logging.info("=" * 70)
    logging.info(f"\nConfiguration:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")
    logging.info(f"\nOutput directory: {run_dir}")
    logging.info(f"Device: {device}")
    logging.info("=" * 70 + "\n")

    # Data loading
    logging.info("Loading data...")
    train_loader, val_loader, val_collate_fn = get_dataloaders(args)
    
    # Model initialization
    logging.info(f"\nLoading model: {args.model}")
    model = get_model(
        args.model, 
        freeze_backbone=args.freeze_backbone,
        vit_model_path=args.vit_model_path
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    logging.info(f"Optimizer: {args.optimizer}")
    
    # Scheduler
    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience
        )
        logging.info(f"Scheduler: ReduceLROnPlateau (patience={args.scheduler_patience})")
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        logging.info("Scheduler: CosineAnnealingLR")
    else:
        scheduler = None
        logging.info("Scheduler: None")

    # Loss and training utilities
    criterion = nn.MSELoss()
    logging.info("Loss function: MSELoss")
    
    scaler = GradScaler() if args.use_amp else None
    if args.use_amp:
        logging.info("Using automatic mixed precision (AMP)")
    
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta
        )
        logging.info(f"Early stopping enabled (patience={args.early_stopping_patience})")

    # Resume from checkpoint
    start_epoch = 0
    best_val_metric = float('inf')
    
    if args.resume and os.path.isfile(args.resume):
        logging.info(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_metric = checkpoint.get('best_val_metric', float('inf'))
            logging.info(f"Resumed from epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            logging.info("Loaded model weights")

    # Training history
    history = {'epoch': [], 'train_loss': [], 'val_mae': [], 'val_mse': [], 'val_r2': [], 'lr': []}

    # Training loop
    logging.info("\n" + "=" * 70)
    logging.info("Starting training")
    logging.info("=" * 70 + "\n")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        logging.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logging.info("-" * 70)
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.model,
            use_amp=args.use_amp, gradient_clip=args.gradient_clip, scaler=scaler
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device, args.model)
        val_mae = val_metrics["mae"]
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        logging.info(f"Epoch {epoch+1} Results:")
        logging.info(f"  Train Loss: {train_loss:.4f}")
        logging.info(f"  Val MAE: {val_mae:.4f}")
        logging.info(f"  Val MSE: {val_metrics['mse']:.4f}")
        logging.info(f"  Val R2: {val_metrics['r2']:.4f}")
        logging.info(f"  LR: {current_lr:.2e}")
        logging.info(f"  Time: {epoch_time:.2f}s")
        
        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_mae'].append(val_mae)
        history['val_mse'].append(val_metrics['mse'])
        history['val_r2'].append(val_metrics['r2'])
        history['lr'].append(current_lr)
        
        # Scheduler step
        if scheduler:
            if args.scheduler == "plateau":
                scheduler.step(val_mae)
            else:
                scheduler.step()
            
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < current_lr:
                logging.info(f"  LR reduced: {current_lr:.2e} -> {new_lr:.2e}")
        
        # Save best model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_metric': best_val_metric,
            'args': vars(args)
        }
        
        if val_mae < best_val_metric:
            best_val_metric = val_mae
            checkpoint['best_val_metric'] = best_val_metric
            torch.save(checkpoint, os.path.join(run_dir, "best_checkpoint.pth"))
            logging.info(f"  New best model! MAE: {best_val_metric:.4f}")
        
        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint, os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            logging.info(f"  Checkpoint saved: epoch_{epoch+1}.pth")
        
        # Early stopping
        if early_stopping and early_stopping(val_mae):
            logging.info(f"\nEarly stopping triggered! Best MAE: {best_val_metric:.4f}")
            break
    
    # Training complete
    logging.info("\n" + "=" * 70)
    logging.info("Training complete!")
    logging.info("=" * 70)
    logging.info(f"Best validation MAE: {best_val_metric:.4f}")
    
    # Save history
    pd.DataFrame(history).to_csv(os.path.join(run_dir, 'training_history.csv'), index=False)
    logging.info(f"Training history saved: {os.path.join(run_dir, 'training_history.csv')}")
    
    # Final test evaluation
    if args.test_annotations and os.path.exists(args.test_annotations):
        logging.info("\n" + "=" * 70)
        logging.info("Final Test Evaluation")
        logging.info("=" * 70)
        
        # Load best model
        best_ckpt = torch.load(os.path.join(run_dir, "best_checkpoint.pth"), 
                               map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        
        # Create test loader
        test_df = pd.read_csv(args.test_annotations)
        logging.info(f"Test size: {len(test_df)}")
        
        test_dataset = RegressionDataset(
            test_df, args.test_image_dir or args.val_image_dir,
            args.filename_col, args.label_col, args.image_size
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=get_num_workers(args.num_workers), pin_memory=True,
            collate_fn=val_collate_fn
        )
        
        # Evaluate
        test_metrics = validate(model, test_loader, device, args.model)
        
        logging.info(f"\nTest Results:")
        logging.info(f"  MAE: {test_metrics['mae']:.4f}")
        logging.info(f"  MSE: {test_metrics['mse']:.4f}")
        logging.info(f"  R2:  {test_metrics['r2']:.4f}")
        
        # Save test results
        with open(os.path.join(run_dir, 'test_results.txt'), 'w') as f:
            f.write("Test Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"test_mae: {test_metrics['mae']:.4f}\n")
            f.write(f"test_mse: {test_metrics['mse']:.4f}\n")
            f.write(f"test_r2: {test_metrics['r2']:.4f}\n")
            f.write(f"best_val_mae: {best_val_metric:.4f}\n")
        
        logging.info(f"Test results saved: {os.path.join(run_dir, 'test_results.txt')}")
    
    logging.info("\n" + "=" * 70)
    logging.info("All tasks complete!")
    logging.info("=" * 70)


if __name__ == '__main__':
    main()
