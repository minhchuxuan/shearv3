#!/usr/bin/env python3
"""
Clean BGE-M3 Pruning Training Script
Minimal implementation with HuggingFace datasets
"""

import torch
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Fix datasets import conflict by importing before local modules
import datasets as hf_datasets

from composer import Trainer
from composer.utils import reproducibility
from omegaconf import OmegaConf

# Import BGE components
from models.composer_bge_m3 import ComposerBGEM3
from data_loader import create_dataloader
from callbacks.pruning_callback import PruningCallback

def setup_optimizer(model, cfg):
    """Setup three-group optimizer for L0 pruning"""
    if not hasattr(model, 'l0_module'):
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Three parameter groups
    model_params = [p for n, p in model.named_parameters() 
                   if not n.startswith('l0_module') and p.requires_grad]
    mask_params = [p for p in model.l0_module.masks.parameters() if p.requires_grad]
    lambda_params = [p for p in model.l0_module.lambdas.parameters() if p.requires_grad]
    
    param_groups = [
        {'params': model_params, 'lr': cfg.lr, 'weight_decay': cfg.weight_decay},
        {'params': mask_params, 'lr': cfg.lr * 10, 'weight_decay': 0.0},
        {'params': lambda_params, 'lr': cfg.lr * 100, 'weight_decay': 0.0}
    ]
    
    return torch.optim.AdamW(param_groups, betas=cfg.get('betas', [0.9, 0.999]), eps=cfg.get('eps', 1e-8))

def main():
    parser = argparse.ArgumentParser(description='BGE-M3 Pruning Training')
    parser.add_argument('config', type=str, help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, default='sts', choices=['sts', 'msmarco'], 
                       help='Dataset to use (sts or msmarco)')
    
    args = parser.parse_args()
    
    # Set seed
    reproducibility.seed_all(args.seed)
    
    # Load configuration
    cfg = OmegaConf.load(args.config)
    print(f"🔥 BGE-M3 Pruning Training")
    print(f"📁 Config: {args.config}")
    print(f"📊 Dataset: {args.dataset}")
    
    # Setup model
    print("🚀 Loading BGE-M3 with L0 pruning...")
    model = ComposerBGEM3(cfg.model)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Total parameters: {total_params:,}")
    print(f"📈 Trainable parameters: {trainable_params:,}")
    
    # Setup data
    print(f"📊 Loading {args.dataset} dataset from HuggingFace...")
    train_dataloader = create_dataloader(
        dataset_name=args.dataset,
        split="train",
        batch_size=cfg.get('batch_size', 16),
        max_length=cfg.get('max_length', 512)
    )
    
    eval_dataloader = create_dataloader(
        dataset_name=args.dataset,
        split="validation",
        batch_size=cfg.get('batch_size', 16),
        max_length=cfg.get('max_length', 512)
    )
    
    # Setup optimizer
    optimizer = setup_optimizer(model, cfg.optimizer)
    
    # Setup callbacks
    callbacks = [PruningCallback(log_interval=500)]
    
    # Convert PyTorch device names to Composer device names
    device = cfg.get('device', 'gpu')  # Default to 'gpu' for Composer
    
    # If user specified 'cuda' or torch.cuda.is_available(), use 'gpu'
    if device == 'cuda' or (device == 'auto' and torch.cuda.is_available()):
        device = 'gpu'
    elif device == 'cpu':
        device = 'cpu'
    else:
        # Default fallback
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        save_interval=cfg.get('save_interval', '500ba'),
        save_folder=cfg.get('save_folder', 'experiments/checkpoints'),
        device=device,
        callbacks=callbacks,
        optimizers=optimizer,
        precision=cfg.get('precision', 'amp_bf16'),
    )
    # Start training
    print("🎯 Starting BGE-M3 pruning training...")
    trainer.fit()
    
    # Print final results
    final_params = sum(p.numel() for p in model.parameters())
    reduction = (total_params - final_params) / total_params * 100
    print(f"✅ Training completed!")
    print(f"📉 Parameter reduction: {reduction:.1f}%")
    print(f"💾 Final model size: {final_params:,} parameters")

if __name__ == "__main__":
    main()
