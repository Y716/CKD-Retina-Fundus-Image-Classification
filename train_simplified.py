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

from model.classifier.simplified_model import ClassificationModel
from model.classifier.dataset import FundusDataModule
from model.losses.losses import FocalLoss, CELoss

console = Console()

def get_loss_function(loss_name, **kwargs):
    """Return the appropriate loss function based on name"""
    if loss_name == "focal":
        return FocalLoss(**kwargs)
    elif loss_name == "ce":
        return CELoss(**kwargs)
    else:
        return nn.CrossEntropyLoss(**kwargs)

class CustomProgressBar(TQDMProgressBar):
    """Custom progress bar with more informative output"""
    def __init__(self):
        super().__init__()
        self.enable = True
        
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        console.print(f"[bold cyan]Starting Epoch {trainer.current_epoch+1}/{trainer.max_epochs}[/bold cyan]")
    
    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        metrics = trainer.callback_metrics
        
        # Create a table for metrics
        table = Table(title=f"Epoch {trainer.current_epoch+1} Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Only display the essential metrics
        essential_metrics = ["train_loss", "train_acc", "val_loss", "val_acc"]
        
        for name, value in metrics.items():
            if name in essential_metrics:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                table.add_row(name, f"{value:.4f}")
            
        console.print(table)

class WandBMetricsCallback(pl.Callback):
    """Custom callback to configure WandB logging behavior"""
    def __init__(self, log_confusion_matrix=False):
        super().__init__()
        self.log_confusion_matrix = log_confusion_matrix
        
    def on_fit_start(self, trainer, pl_module):
        """Set up WandB to track only essential metrics in one graph"""
        if trainer.logger and isinstance(trainer.logger, WandbLogger):
            # Configure which metrics to track and how to display them
            trainer.logger.experiment.define_metric("train_loss", summary="min")
            trainer.logger.experiment.define_metric("train_acc", summary="max")
            trainer.logger.experiment.define_metric("val_loss", summary="min")
            trainer.logger.experiment.define_metric("val_acc", summary="max")
            
            # Set up a custom chart to compare the metrics
            trainer.logger.experiment.define_metric("custom_step")
            for metric in ["train_loss", "val_loss", "train_acc", "val_acc"]:
                trainer.logger.experiment.define_metric(
                    metric, 
                    step_metric="custom_step",
                    goal="minimize" if "loss" in metric else "maximize"
                )
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log custom step to group metrics together"""
        if trainer.logger and isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log({"custom_step": trainer.current_epoch})

def main():
    # Set up argument parser with more options
    parser = argparse.ArgumentParser(description="Fundus Image Classification Training")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="efficientnet_b0",
                        help="Model architecture from timm (e.g., efficientnet_b0, resnet50)")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of classes to predict")
    parser.add_argument("--in_channels", type=int, default=3,
                        help="Number of input channels")
    parser.add_argument("--pretrained", action="store_true", 
                        help="Use pretrained weights")
    
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
                        help="Loss function to use (ce=CrossEntropy, focal=FocalLoss)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma parameter for focal loss")
    
    # Logging and checkpointing
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom name for this run")
    parser.add_argument("--tags", type=str, default=None,
                        help="Comma-separated tags for this run")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--upload_models", action="store_true",
                        help="Upload model checkpoints to WandB (uses more storage)")
    parser.add_argument("--save_top_k", type=int, default=1,
                        help="Number of best models to save")
    parser.add_argument("--wandb_mode", type=str, default="online", 
                        choices=["online", "offline", "disabled"],
                        help="WandB logging mode")
    parser.add_argument("--log_confusion_matrix", action="store_true",
                        help="Log confusion matrix at the end of training")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Print training configuration
    console.print("[bold green]Training Configuration:[/bold green]")
    config_table = Table()
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="yellow")
    
    for arg, value in vars(args).items():
        config_table.add_row(arg, str(value))
    
    console.print(config_table)
    
    # Generate run name if not provided
    if args.run_name is None:
        # Create a unique ID
        unique_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.run_name = f"{args.model_name}_{timestamp}_{unique_id}"
    
    # Parse tags
    tags = None
    if args.tags:
        tags = [tag.strip() for tag in args.tags.split(",")]
    
    # Set WandB mode
    os.environ["WANDB_MODE"] = args.wandb_mode
    
    # Initialize WandB logger with custom settings
    wandb_logger = WandbLogger(
        project="fundus-classification",
        name=args.run_name,
        tags=tags,
        log_model="all" if args.upload_models else None,  # Only upload models if requested
        config=vars(args)
    )
    
    # Initialize loss function
    if args.loss == "focal":
        loss_fn = FocalLoss(gamma=args.focal_gamma)
        console.print(f"[bold]Using Focal Loss with gamma={args.focal_gamma}[/bold]")
    else:
        loss_fn = CELoss()
        console.print("[bold]Using Cross Entropy Loss[/bold]")
    
    # Initialize model
    console.print(f"[bold]Initializing {args.model_name} model...[/bold]")
    model = ClassificationModel(
        model_name=args.model_name,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        loss_fn=loss_fn,
        lr=args.lr,
        weight_decay=args.weight_decay,
        simplified_logging=True  # Use simplified logging
    )
    
    # Print model summary
    console.print(f"[bold]Model Summary:[/bold]")
    console.print(f"Architecture: {args.model_name}")
    console.print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize data module
    console.print("[bold]Setting up data loaders...[/bold]")
    datamodule = FundusDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # Setup callbacks
    callbacks = [
        CustomProgressBar(),
        EarlyStopping(
            monitor="val_loss", 
            patience=args.early_stopping, 
            mode="min",
            verbose=True
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_dir, args.run_name),
            filename="{epoch}-{val_loss:.4f}-{val_acc:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=args.save_top_k,
            verbose=True
        ),
        WandBMetricsCallback(log_confusion_matrix=args.log_confusion_matrix)
    ]
    
    # Initialize trainer
    console.print("[bold]Initializing PyTorch Lightning Trainer...[/bold]")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=50,  # Less frequent logging
        accelerator="auto",
        devices=1,
        strategy="auto"
    )
    
    # Start training
    console.print("[bold green]Starting training...[/bold green]")
    trainer.fit(model, datamodule=datamodule)
    
    # Print final results
    console.print("[bold green]Training completed![/bold green]")
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_val_score = trainer.checkpoint_callback.best_model_score.item()
    
    console.print(f"[bold]Best model saved at:[/bold] {best_model_path}")
    console.print(f"[bold]Best validation loss:[/bold] {best_val_score:.4f}")
    
    # Log the best model path to wandb
    wandb.log({"best_model_path": best_model_path, "best_val_score": best_val_score})
    
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()