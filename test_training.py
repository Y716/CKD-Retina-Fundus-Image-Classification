#!/usr/bin/env python
"""
Quick test script to verify training works before running experiments
"""

import subprocess
import sys
import time
from rich.console import Console

console = Console()

def test_training():
    """Run a quick training test with minimal epochs"""
    console.print("[bold cyan]Running training test...[/bold cyan]")
    
    # Test command with only 3 epochs
    cmd = [
        "python", "train.py",
        "--model_name=mobilenetv2_100",
        "--batch_size=32",
        "--lr=0.001",
        "--max_epochs=3",
        "--wandb_mode=offline",  # Don't upload to wandb for test
        "--run_name=test_run"
    ]
    
    console.print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Capture output
        output_lines = []
        for line in process.stdout:
            print(line.strip())
            output_lines.append(line.strip())
        
        return_code = process.wait()
        elapsed = time.time() - start_time
        
        if return_code == 0:
            console.print(f"\n[bold green]✓ Test completed successfully in {elapsed:.1f}s![/bold green]")
            console.print("Training script is working properly.")
            return True
        else:
            console.print(f"\n[bold red]✗ Test failed with return code {return_code}[/bold red]")
            console.print("Last 20 lines of output:")
            for line in output_lines[-20:]:
                console.print(f"  {line}")
            return False
            
    except Exception as e:
        console.print(f"\n[bold red]✗ Test failed with exception: {e}[/bold red]")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    console.print("[bold cyan]Checking dependencies...[/bold cyan]")
    
    issues = []
    
    # Check Python packages
    required_packages = [
        "torch",
        "torchvision",
        "lightning",
        "wandb",
        "timm",
        "rich",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            console.print(f"[green]✓ {package}[/green]")
        except ImportError:
            console.print(f"[red]✗ {package} not installed[/red]")
            issues.append(f"{package} not installed")
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            console.print(f"[green]✓ CUDA available: {torch.cuda.get_device_name(0)}[/green]")
        else:
            console.print("[yellow]⚠ CUDA not available - will use CPU (slower)[/yellow]")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not check CUDA: {e}[/yellow]")
    
    # Check data directory
    import os
    if os.path.exists("split_dataset/train") and os.path.exists("split_dataset/test"):
        # Count images
        train_count = sum([len(files) for r, d, files in os.walk("split_dataset/train")])
        test_count = sum([len(files) for r, d, files in os.walk("split_dataset/test")])
        console.print(f"[green]✓ Dataset found: {train_count} train, {test_count} test images[/green]")
    else:
        console.print("[red]✗ Dataset not found in split_dataset/[/red]")
        issues.append("Dataset not found")
    
    return len(issues) == 0

def main():
    console.print("[bold]Training System Test[/bold]\n")
    
    # Check dependencies first
    if not check_dependencies():
        console.print("\n[bold red]Please fix the issues above before running experiments.[/bold red]")
        sys.exit(1)
    
    console.print("")
    
    # Run training test
    if test_training():
        console.print("\n[bold green]All tests passed! You can now run experiments.[/bold green]")
        console.print("\nTry: python run_experiments.py --config experiments/01_benchmark_models.json")
    else:
        console.print("\n[bold red]Training test failed. Please check the error messages above.[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()