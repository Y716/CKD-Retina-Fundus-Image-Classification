#!/usr/bin/env python
"""
Simple test to verify the exact issue
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from rich.console import Console

console = Console()

def test_data_loading():
    """Test basic data loading"""
    console.print("[bold cyan]Testing data loading...[/bold cyan]")
    
    # Simple transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset directly
    try:
        train_dataset = datasets.ImageFolder("split_dataset/train", transform=transform)
        console.print(f"[green]✓ Loaded training dataset: {len(train_dataset)} samples[/green]")
        console.print(f"  Classes: {train_dataset.classes}")
        
        # Create dataloader
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        
        # Test loading a batch
        for i, data in enumerate(train_loader):
            console.print(f"\n[bold]Batch {i+1}:[/bold]")
            console.print(f"  Type: {type(data)}")
            console.print(f"  Length: {len(data)}")
            
            # Check what's in the batch
            if isinstance(data, (list, tuple)):
                for j, item in enumerate(data):
                    if torch.is_tensor(item):
                        console.print(f"  Item {j}: Tensor with shape {item.shape}")
                    else:
                        console.print(f"  Item {j}: {type(item)}")
                        
                # Try unpacking
                try:
                    images, labels = data
                    console.print(f"\n[green]✓ Standard unpacking works![/green]")
                    console.print(f"  Images: {images.shape}, dtype: {images.dtype}")
                    console.print(f"  Labels: {labels.shape}, dtype: {labels.dtype}")
                    console.print(f"  Label values: {labels.tolist()}")
                except ValueError as e:
                    console.print(f"\n[red]✗ Standard unpacking failed: {e}[/red]")
                    console.print("  This is the issue!")
            
            if i >= 2:  # Check first 3 batches
                break
                
    except Exception as e:
        console.print(f"[red]✗ Error loading dataset: {e}[/red]")
        import traceback
        traceback.print_exc()

def test_model_input():
    """Test if model can process data"""
    console.print("\n[bold cyan]Testing model input...[/bold cyan]")
    
    import timm
    
    # Create a simple model
    model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=5)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(2, 3, 224, 224)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        console.print(f"[green]✓ Model processes input correctly[/green]")
        console.print(f"  Input shape: {dummy_input.shape}")
        console.print(f"  Output shape: {output.shape}")
    except Exception as e:
        console.print(f"[red]✗ Model failed: {e}[/red]")

if __name__ == "__main__":
    test_data_loading()
    test_model_input()