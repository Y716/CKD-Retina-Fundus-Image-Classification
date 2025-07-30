#!/usr/bin/env python
"""
Unified training script for fundus image classification.
Default behavior: No model upload to WandB, only essential metrics logged.
"""

import argparse
import os
import time
import wandb
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping, 
    LearningRateMonitor, 
    ModelCheckpoint,
    TQDMProgressBar
)
import random
import string
from rich.console import Console
from rich.table import Table

from model.classifier.model import ClassificationModel
from model.classifier.dataset import FundusDataModule
from model.losses.losses import FocalLoss, CELoss

console = Console()

def get_loss_function(loss_name, **kwargs):
    """Return the appropriate loss function based on name"""
    if loss_name == "focal":
        # Only pass gamma to FocalLoss
        focal_kwargs = {k: v for k, v in kwargs.items() if k in ['gamma', 'weight', 'reduction']}
        return FocalLoss(**focal_kwargs)
    elif loss_name == "ce":
        # Only pass relevant args to CELoss
        ce_kwargs = {k: v for k, v in kwargs.items() if k in ['weight', 'ignore_index', 'reduction']}
        return CELoss(**ce_kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

class MinimalProgressBar(TQDMProgressBar):
    """Minimal progress bar that only shows essential information"""
    def __init__(self):
        super().__init__()
        self.enable = True
        
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        console.print(f"[bold cyan]Epoch {trainer.current_epoch+1}/{trainer.max_epochs}[/bold cyan]")
    
    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        try:
            metrics = trainer.callback_metrics
            
            # Only show essential metrics
            essential_metrics = {}
            for key in ["train_loss", "train_acc", "val_loss", "val_acc"]:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    essential_metrics[key] = value
            
            # Create a compact display
            if essential_metrics:
                metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in essential_metrics.items()])
                console.print(f"[green]{metric_str}[/green]")
        except Exception as e:
            # Silent fail to not interrupt training
            pass

class WandBEssentialsLogger(WandbLogger):
    """Custom WandB logger that only logs essential metrics"""
    def log_metrics(self, metrics, step):
        # Filter to only essential metrics
        essential_keys = ["train_loss", "train_acc", "val_loss", "val_acc", "epoch"]
        filtered_metrics = {k: v for k, v in metrics.items() if k in essential_keys}
        
        if filtered_metrics:
            super().log_metrics(filtered_metrics, step)

def main():
    parser = argparse.ArgumentParser(description="Fundus Image Classification Training")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="efficientnet_b0",
                        help="Model architecture from timm")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of classes to predict")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights (default: True)")
    
    # Data parameters
    parser.add_argument("--train_dir", type=str, default="split_dataset/train",
                        help="Directory containing training data")
    parser.add_argument("--val_dir", type=str, default="split_dataset/test",
                        help="Directory containing validation data")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and validation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for optimizer")
    parser.add_argument("--max_epochs", type=int, default=30,
                        help="Maximum number of training epochs")
    parser.add_argument("--early_stopping", type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"],
                        help="Loss function to use")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma parameter for focal loss")
    
    # Optimizer and scheduler
    parser.add_argument("--optimizer", type=str, default="adamw", 
                        choices=["adam", "adamw", "sgd"],
                        help="Optimizer to use")
    parser.add_argument("--scheduler", type=str, default=None,
                        choices=[None, "cosine", "reduce_on_plateau", "step"],
                        help="Learning rate scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_factor", type=float, default=0.1,
                        help="Factor for ReduceLROnPlateau scheduler")
    
    # Logging and checkpointing
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for this experiment")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom name for this run")
    parser.add_argument("--tags", type=str, default=None,
                        help="Comma-separated tags for this run")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_top_k", type=int, default=1,
                        help="Number of best models to save")
    parser.add_argument("--wandb_project", type=str, default="fundus-classification",
                        help="WandB project name")
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"],
                        help="WandB logging mode")
    parser.add_argument("--upload_models", action="store_true", default=False,
                        help="Upload model checkpoints to WandB (default: False)")
    parser.add_argument("--log_confusion_matrix", action="store_true", default=False,
                        help="Log confusion matrix (default: False)")
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Generate run name if not provided
    if args.run_name is None:
        unique_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        args.run_name = f"{args.model_name}_{unique_id}"
    
    # Add experiment name to run name if provided
    if args.experiment_name:
        args.run_name = f"{args.experiment_name}_{args.run_name}"
    
    # Parse tags
    tags = []
    if args.tags:
        tags = [tag.strip() for tag in args.tags.split(",")]
    if args.experiment_name:
        tags.append(args.experiment_name)
    
    # Print configuration
    console.print("[bold green]Training Configuration:[/bold green]")
    console.print(f"Model: {args.model_name}")
    console.print(f"Batch size: {args.batch_size}")
    console.print(f"Learning rate: {args.lr}")
    console.print(f"Loss: {args.loss}")
    console.print(f"Epochs: {args.max_epochs}")
    console.print(f"Run name: {args.run_name}")
    
    # Set WandB mode
    os.environ["WANDB_MODE"] = args.wandb_mode
    
    # Initialize WandB logger
    wandb_logger = WandBEssentialsLogger(
        project=args.wandb_project,
        name=args.run_name,
        tags=tags,
        log_model="all" if args.upload_models else None,
        config=vars(args)
    )
    
    # Initialize loss function
    if args.loss == "focal":
        loss_fn = get_loss_function(args.loss, gamma=args.focal_gamma)
    else:
        loss_fn = get_loss_function(args.loss)
    
    # Prepare scheduler parameters
    scheduler_params = {}
    if args.scheduler == "reduce_on_plateau":
        scheduler_params = {
            "patience": args.scheduler_patience,
            "factor": args.scheduler_factor,
            "mode": "min"
        }
    elif args.scheduler == "cosine":
        scheduler_params = {
            "T_max": args.max_epochs,
            "eta_min": 1e-6
        }
    elif args.scheduler == "step":
        scheduler_params = {
            "step_size": 10,
            "gamma": 0.1
        }
    
    # Initialize model with minimal logging
    model = ClassificationModel(
        model_name=args.model_name,
        num_classes=args.num_classes,
        in_channels=3,
        loss_fn=loss_fn,
        metric_fn=None,  # Disable extra metrics by default
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        scheduler_params=scheduler_params
    )
    
    # Override the validation epoch end to disable confusion matrix by default
    if not args.log_confusion_matrix:
        model.on_validation_epoch_end = lambda: None
    
    # Initialize data module
    datamodule = FundusDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # Setup callbacks
    callbacks = [
        MinimalProgressBar(),
        EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping,
            mode="min",
            verbose=False
        ),
        ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_dir, args.run_name),
            filename="{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=args.save_top_k,
            verbose=False
        )
    ]
    
    # Only add LR monitor if using a scheduler
    if args.scheduler:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=50,  # Less frequent logging
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False,  # Disable model summary
        gradient_clip_val=1.0,  # Add gradient clipping for stability
        deterministic=True  # For reproducibility
    )
    
    # Start training
    console.print("[bold green]Starting training...[/bold green]")
    try:
        trainer.fit(model, datamodule=datamodule)
        
        # Print final results
        console.print("[bold green]Training completed![/bold green]")
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_val_loss = trainer.checkpoint_callback.best_model_score.item() if trainer.checkpoint_callback.best_model_score else None
        
        if best_val_loss:
            console.print(f"Best validation loss: {best_val_loss:.4f}")
            console.print(f"Best model saved at: {best_model_path}")
        
        # Log final metrics
        if wandb.run is not None:
            wandb.log({
                "best_model_path": best_model_path,
                "best_val_loss": best_val_loss
            })
        
    except Exception as e:
        console.print(f"[bold red]Training failed with error: {e}[/bold red]")
        raise
    finally:
        # Finish WandB run
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()