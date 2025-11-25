"""
Helper utilities for Vesuvius Challenge.

This module provides common utility functions for training,
evaluation, and data processing.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    # If using PyTorch:
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def save_checkpoint(
    model: Any,
    optimizer: Any,
    epoch: int,
    loss: float,
    path: str,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        loss: Current loss value.
        path: Path to save the checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state": None,  # model.state_dict() for PyTorch
        "optimizer_state": None,  # optimizer.state_dict() for PyTorch
        "loss": loss,
    }
    
    # Placeholder: Implement actual saving
    # torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        path: Path to the checkpoint file.
        
    Returns:
        Checkpoint dictionary.
    """
    # Placeholder: Implement actual loading
    # return torch.load(path)
    return {}


def dice_coefficient(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """Calculate Dice coefficient for binary segmentation.
    
    Args:
        pred: Predicted binary mask.
        target: Ground truth binary mask.
        smooth: Smoothing factor to avoid division by zero.
        
    Returns:
        Dice coefficient value between 0 and 1.
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat)
    
    return (2.0 * intersection + smooth) / (union + smooth)


def fbeta_score(
    pred: np.ndarray,
    target: np.ndarray,
    beta: float = 0.5,
    threshold: float = 0.5,
) -> float:
    """Calculate F-beta score (competition metric).
    
    The Vesuvius Challenge uses F0.5 score which weighs precision
    more heavily than recall.
    
    Args:
        pred: Predicted probabilities.
        target: Ground truth binary mask.
        beta: Beta parameter for F-score (0.5 for competition).
        threshold: Threshold for binarizing predictions.
        
    Returns:
        F-beta score value.
    """
    pred_binary = (pred > threshold).astype(np.float32)
    
    tp = np.sum(pred_binary * target)
    fp = np.sum(pred_binary * (1 - target))
    fn = np.sum((1 - pred_binary) * target)
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    beta_squared = beta ** 2
    fbeta = (1 + beta_squared) * precision * recall / (
        beta_squared * precision + recall + 1e-6
    )
    
    return fbeta


def normalize_volume(
    volume: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> np.ndarray:
    """Normalize a 3D volume.
    
    Args:
        volume: Input volume array.
        mean: Mean for normalization. If None, computed from volume.
        std: Std for normalization. If None, computed from volume.
        
    Returns:
        Normalized volume.
    """
    if mean is None:
        mean = np.mean(volume)
    if std is None:
        std = np.std(volume)
    
    return (volume - mean) / (std + 1e-6)


def get_device() -> str:
    """Get the available device for computation.
    
    Returns:
        Device string ('cuda' or 'cpu').
    """
    # Placeholder: Check for GPU availability
    # import torch
    # return 'cuda' if torch.cuda.is_available() else 'cpu'
    return "cpu"


def count_parameters(model: Any) -> int:
    """Count trainable parameters in a model.
    
    Args:
        model: Model instance.
        
    Returns:
        Number of trainable parameters.
    """
    # For PyTorch models:
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # For placeholder models with get_params_count method:
    if hasattr(model, "get_params_count"):
        return model.get_params_count()
    # TODO: Implement actual parameter counting when using real PyTorch models
    return 0


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """Update statistics with new value.
        
        Args:
            val: New value to add.
            n: Count of values (for batch updates).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' for loss, 'max' for metrics like accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score.
            
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
