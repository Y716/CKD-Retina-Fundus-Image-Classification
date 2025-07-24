#!/usr/bin/env python
"""
This script performs a simple test run of the model and data pipeline to identify issues
without running a full training cycle.
"""

import argparse
import torch
import torch.nn as nn
from rich.console import Console
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

from model.classifier.model import ClassificationModel
from model.classifier.dataset import FundusDataModule
from model.losses.losses import CELoss

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Debug fundus classification model")
    parser.add_argument("--train_dir", type=str, default="split_dataset/train",
                        help="Directory containing training data")
    parser.add_argument("--val_dir", type=str, default="split_dataset/test",
                        help="Directory containing validation data")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for testing (use small value)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--model_name", type=str, default="efficientnet_b0",
                        help="Model architecture from timm")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of classes to predict")
    parser.add_argument("--output_dir", type=str, default="debug_output",
                        help="Directory to save debug outputs")
    return parser.parse_args()

def test_datamodule(datamodule, output_dir):
    """Test the datamodule to ensure it can load images properly"""
    console.print("[bold cyan]Testing DataModule...[/bold cyan]")
    
    # Setup the datamodule
    try:
        datamodule.setup()
        console.print("[green]✓ DataModule setup successful[/green]")
    except Exception as e:
        console.print(f"[red]✗ DataModule setup failed: {e}[/red]")
        raise
    
    # Check if we can get a batch from the dataloader
    try:
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        console.print("[green]✓ Successfully loaded a batch from train_dataloader[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to load a batch from train_dataloader: {e}[/red]")
        raise
    
    # Check batch structure
    try:
        images, labels = batch
        console.print(f"[green]✓ Batch structure is correct: images shape {images.shape}, labels shape {labels.shape}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Batch structure is incorrect: {e}[/red]")
        raise
    
    # Check image values
    try:
        console.print(f"Image tensor statistics: min={images.min().item():.4f}, max={images.max().item():.4f}, mean={images.mean().item():.4f}")
        if images.min() < -5 or images.max() > 5:
            console.print("[yellow]⚠ Image values seem unusual. Check normalization.[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ Failed to check image statistics: {e}[/red]")
    
    # Check label values
    try:
        unique_labels = torch.unique(labels)
        console.print(f"Unique label values: {unique_labels.tolist()}")
        if len(unique_labels) < 2:
            console.print("[yellow]⚠ Only one class detected in this batch. This might be normal for a small batch.[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ Failed to check label values: {e}[/red]")
    
    # Visualize a few images
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Denormalize images for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images_vis = images * std + mean
        
        # Clip values to valid range
        images_vis = torch.clamp(images_vis, 0, 1)
        
        # Plot images
        fig, axes = plt.subplots(1, min(4, images.shape[0]), figsize=(15, 5))
        if images.shape[0] == 1:
            axes = [axes]
            
        for i, ax in enumerate(axes):
            if i < images.shape[0]:
                img = images_vis[i].permute(1, 2, 0).cpu().numpy()
                ax.imshow(img)
                ax.set_title(f"Label: {labels[i].item()}")
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sample_images.png"))
        console.print(f"[green]✓ Saved sample images to {os.path.join(output_dir, 'sample_images.png')}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to visualize images: {e}[/red]")

def test_model(model, datamodule, output_dir):
    """Test the model to ensure it can process images and compute loss"""
    console.print("[bold cyan]Testing Model...[/bold cyan]")
    
    # Load a batch
    try:
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        images, labels = batch
        console.print("[green]✓ Successfully loaded a batch for model testing[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to load a batch for model testing: {e}[/red]")
        raise
    
    # Test forward pass
    try:
        with torch.no_grad():
            outputs = model(images)
        console.print(f"[green]✓ Forward pass successful: outputs shape {outputs.shape}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Forward pass failed: {e}[/red]")
        raise
    
    # Test loss computation
    try:
        loss = model.loss_fn(outputs, labels)
        console.print(f"[green]✓ Loss computation successful: loss value {loss.item():.4f}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Loss computation failed: {e}[/red]")
        raise
    
    # Test predictions
    try:
        preds = torch.argmax(outputs, dim=1)
        console.print(f"Predictions: {preds.tolist()}")
        console.print(f"Ground truth: {labels.tolist()}")
    except Exception as e:
        console.print(f"[red]✗ Failed to compute predictions: {e}[/red]")
    
    # Test training step
    try:
        loss = model.training_step(batch, 0)
        console.print(f"[green]✓ Training step successful: loss value {loss.item():.4f}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Training step failed: {e}[/red]")
        raise
    
    # Test validation step
    try:
        result = model.validation_step(batch, 0)
        console.print(f"[green]✓ Validation step successful: {result.keys()}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Validation step failed: {e}[/red]")
        raise

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    console.print("[bold]Starting debug process...[/bold]")
    
    # Initialize datamodule
    console.print("[bold]Initializing DataModule...[/bold]")
    datamodule = FundusDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=0  # Use 0 for easier debugging
    )
    
    # Test datamodule
    test_datamodule(datamodule, args.output_dir)
    
    # Initialize model
    console.print("[bold]Initializing Model...[/bold]")
    model = ClassificationModel(
        model_name=args.model_name,
        num_classes=args.num_classes,
        in_channels=3,
        loss_fn=CELoss()
    )
    
    # Test model
    test_model(model, datamodule, args.output_dir)
    
    console.print("[bold green]Debug process completed successfully![/bold green]")
    console.print("If you didn't see any errors, your model and datamodule are working correctly.")
    console.print("You can now try running the full training script.")

if __name__ == "__main__":
    main()