#!/usr/bin/env python
"""
Debug script to check dataset loading
"""

import torch
from model.classifier.dataset import FundusDataModule
from rich.console import Console

console = Console()

def debug_dataset():
    """Check what the dataset is returning"""
    console.print("[bold cyan]Debugging dataset...[/bold cyan]")
    
    # Create data module
    datamodule = FundusDataModule(
        train_dir="split_dataset/train",
        val_dir="split_dataset/test",
        batch_size=4,
        img_size=224,
        num_workers=0  # Use 0 for debugging
    )
    
    # Setup
    datamodule.setup()
    
    # Get dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    console.print("\n[bold]Checking training dataloader:[/bold]")
    # Check first batch
    for i, batch in enumerate(train_loader):
        console.print(f"Batch type: {type(batch)}")
        console.print(f"Batch length: {len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}")
        
        if isinstance(batch, (list, tuple)):
            for j, item in enumerate(batch):
                console.print(f"  Item {j}: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'N/A'}")
                if hasattr(item, 'shape'):
                    console.print(f"  Item {j} dtype: {item.dtype}")
                    console.print(f"  Item {j} min/max: {item.min():.3f}/{item.max():.3f}")
        
        # Try unpacking
        try:
            if len(batch) == 2:
                images, labels = batch
                console.print(f"\n[green]✓ Successfully unpacked batch[/green]")
                console.print(f"  Images shape: {images.shape}")
                console.print(f"  Labels shape: {labels.shape}")
                console.print(f"  Unique labels in batch: {torch.unique(labels).tolist()}")
            else:
                console.print(f"\n[red]✗ Batch has {len(batch)} elements, expected 2[/red]")
        except Exception as e:
            console.print(f"\n[red]✗ Error unpacking batch: {e}[/red]")
        
        break  # Just check first batch
    
    console.print("\n[bold]Checking validation dataloader:[/bold]")
    # Check validation batch
    for i, batch in enumerate(val_loader):
        console.print(f"Batch type: {type(batch)}")
        console.print(f"Batch length: {len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}")
        
        try:
            if len(batch) == 2:
                images, labels = batch
                console.print(f"\n[green]✓ Successfully unpacked validation batch[/green]")
                console.print(f"  Images shape: {images.shape}")
                console.print(f"  Labels shape: {labels.shape}")
            else:
                console.print(f"\n[red]✗ Validation batch has {len(batch)} elements, expected 2[/red]")
        except Exception as e:
            console.print(f"\n[red]✗ Error unpacking validation batch: {e}[/red]")
        
        break

if __name__ == "__main__":
    debug_dataset()