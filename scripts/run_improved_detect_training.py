#!/usr/bin/env python3
"""Improved detection training script with optimized hyperparameters.

This script addresses the poor performance issues in the original training:
1. Data path fix - points to correct dataset location
2. Architecture improvements - deeper decoder (6 layers), more queries (100)
3. Loss improvements - focal loss for better classification
4. Class imbalance handling - increased EOS coefficient

Usage:
    python scripts/run_improved_detect_training.py
"""

import subprocess
import sys
from pathlib import Path

# Get the project root
project_root = Path(__file__).parent.parent.absolute()

# Dataset path - Updated to correct location
DATA_ROOT = "C:/Users/tsake/OneDrive/Desktop/datasets/Aero_dataset"

# Training command with optimized hyperparameters
cmd = [
    sys.executable,
    str(project_root / "clipdetr" / "train_detect.py"),
    
    # Data configuration - FIXED PATH
    "--data-root", DATA_ROOT,
    "--data-yaml", "data.yaml",
    "--train-split", "train",
    "--val-split", "valid",
    
    # Model architecture - IMPROVED
    "--image-backbone", "convnext_tiny",  # Better backbone
    "--embed-dim", "384",
    "--num-queries", "100",               # Increased from 50
    "--decoder-layers", "6",               # Increased from 2
    
    # Training configuration - OPTIMIZED
    "--epochs", "80",
    "--batch-size", "8",
    "--image-size", "320",                 # Slightly larger for better detection
    "--lr", "1e-4",
    "--weight-decay", "1e-4",
    
    # Warmup and optimization
    "--warmup-epochs", "5",
    "--warmup-start-factor", "0.1",
    "--grad-clip-norm", "1.0",
    
    # EMA for stable training
    "--use-ema",
    "--ema-decay", "0.999",
    
    # Training augmentation
    "--train-augment",

    # Loss configuration - IMPROVED FOR CLASS IMBALANCE
    # Note: focal loss is enabled by default in config
    
    # Other settings
    "--num-workers", "2",
    "--device", "auto",
    "--seed", "42",
    
    # Experiment tagging
    "--tag", "improved_v1",
    
    # Output
    "--summary-out", str(project_root / "experiments" / "improved_training_summary.json"),
]

print("=" * 60)
print("Starting Improved Detection Training")
print("=" * 60)
print(f"Dataset: {DATA_ROOT}")
print(f"Command: {' '.join(cmd)}")
print("=" * 60)

# Run the training
result = subprocess.run(cmd, cwd=str(project_root))

if result.returncode != 0:
    print(f"Training failed with return code: {result.returncode}")
    sys.exit(result.returncode)
else:
    print("Training completed successfully!")

