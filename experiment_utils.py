#!/usr/bin/env python
"""
Utility script for managing experiments, analyzing results, and cleaning up.
"""

import argparse
import os
import json
import shutil
import wandb
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
from datetime import datetime

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment management utilities")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List experiments
    list_parser = subparsers.add_parser('list', help='List all experiment configs')
    
    # Clean checkpoints
    clean_parser = subparsers.add_parser('clean', help='Clean up old checkpoints')
    clean_parser.add_argument('--keep-best', type=int, default=1, 
                              help='Number of best checkpoints to keep per experiment')
    clean_parser.add_argument('--dry-run', action='store_true',
                              help='Show what would be deleted without deleting')
    
    # Analyze results
    analyze_parser = subparsers.add_parser('analyze', help='Analyze experiment results from WandB')
    analyze_parser.add_argument('--project', type=str, default='fundus-classification',
                                help='WandB project name')
    analyze_parser.add_argument('--experiment', type=str, default=None,
                                help='Filter by experiment name')
    analyze_parser.add_argument('--export', type=str, default=None,
                                help='Export results to CSV file')
    
    # Create new experiment
    create_parser = subparsers.add_parser('create', help='Create new experiment config')
    create_parser.add_argument('--name', type=str, required=True,
                               help='Experiment name')
    create_parser.add_argument('--template', type=str, default=None,
                               help='Base template to use')
    
    return parser.parse_args()

def list_experiments():
    """List all experiment configurations"""
    exp_dir = "experiments"
    if not os.path.exists(exp_dir):
        console.print("[yellow]No experiments directory found[/yellow]")
        return
    
    configs = [f for f in os.listdir(exp_dir) if f.endswith('.json')]
    
    if not configs:
        console.print("[yellow]No experiment configurations found[/yellow]")
        return
    
    table = Table(title="Available Experiment Configurations")
    table.add_column("Config File", style="cyan")
    table.add_column("Experiment Name", style="green")
    table.add_column("Description", style="yellow")
    table.add_column("# Combinations", style="magenta")
    
    for config_file in sorted(configs):
        config_path = os.path.join(exp_dir, config_file)
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            exp_name = config.get('experiment_name', 'N/A')
            description = config.get('description', 'No description')
            
            # Calculate number of combinations
            if 'parameters' in config:
                n_combos = 1
                for param, values in config['parameters'].items():
                    if isinstance(values, list):
                        n_combos *= len(values)
            else:
                n_combos = 0
            
            table.add_row(config_file, exp_name, description, str(n_combos))
            
        except Exception as e:
            table.add_row(config_file, "Error", str(e), "?")
    
    console.print(table)

def clean_checkpoints(keep_best=1, dry_run=False):
    """Clean up old checkpoints, keeping only the best ones"""
    checkpoint_dir = "checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        console.print("[yellow]No checkpoints directory found[/yellow]")
        return
    
    console.print(f"[bold]Cleaning checkpoints (keeping best {keep_best} per experiment)...[/bold]")
    
    total_size = 0
    files_to_delete = []
    
    # Process each experiment directory
    for exp_dir in os.listdir(checkpoint_dir):
        exp_path = os.path.join(checkpoint_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        
        # Find all checkpoint files
        ckpt_files = [f for f in os.listdir(exp_path) if f.endswith('.ckpt')]
        
        if len(ckpt_files) <= keep_best:
            continue
        
        # Sort by validation loss (assuming format: epoch-val_loss-val_acc.ckpt)
        def get_val_loss(filename):
            try:
                parts = filename.replace('.ckpt', '').split('-')
                return float(parts[1])
            except:
                return float('inf')
        
        ckpt_files.sort(key=get_val_loss)
        
        # Files to delete (all except the best ones)
        to_delete = ckpt_files[keep_best:]
        
        for f in to_delete:
            file_path = os.path.join(exp_path, f)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            files_to_delete.append((file_path, file_size))
    
    if not files_to_delete:
        console.print("[green]No files to delete - all experiments have <= {keep_best} checkpoints[/green]")
        return
    
    # Show what will be deleted
    console.print(f"\n[bold]Found {len(files_to_delete)} files to delete[/bold]")
    console.print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    
    if dry_run:
        console.print("\n[yellow]Dry run - files that would be deleted:[/yellow]")
        for file_path, size in files_to_delete[:10]:  # Show first 10
            console.print(f"  {file_path} ({size / 1024 / 1024:.1f} MB)")
        if len(files_to_delete) > 10:
            console.print(f"  ... and {len(files_to_delete) - 10} more files")
    else:
        if Confirm.ask("\nProceed with deletion?"):
            for file_path, _ in files_to_delete:
                os.remove(file_path)
            console.print(f"[green]✓ Deleted {len(files_to_delete)} files[/green]")

def analyze_results(project, experiment=None, export=None):
    """Analyze experiment results from WandB"""
    console.print(f"[bold]Fetching results from WandB project: {project}[/bold]")
    
    try:
        api = wandb.Api()
        runs = api.runs(project)
        
        if experiment:
            runs = [r for r in runs if experiment in r.tags or experiment in r.name]
            console.print(f"Filtered to {len(runs)} runs with experiment: {experiment}")
        
        if not runs:
            console.print("[yellow]No runs found[/yellow]")
            return
        
        # Collect data
        data = []
        for run in runs:
            if run.state != "finished":
                continue
                
            run_data = {
                "name": run.name,
                "state": run.state,
                "created": run.created_at,
                "duration": run.summary.get("_runtime", 0) / 60,  # in minutes
                "model": run.config.get("model_name", "unknown"),
                "lr": run.config.get("lr", 0),
                "batch_size": run.config.get("batch_size", 0),
                "loss": run.config.get("loss", "unknown"),
                "optimizer": run.config.get("optimizer", "adamw"),
                "best_val_loss": run.summary.get("best_val_loss", float('inf')),
                "val_acc": run.summary.get("val_acc", 0),
                "train_acc": run.summary.get("train_acc", 0),
                "epochs": run.summary.get("epoch", 0),
            }
            
            # Add experiment tag
            for tag in run.tags:
                if tag.startswith("exp_") or "_exploration" in tag or "_comparison" in tag:
                    run_data["experiment"] = tag
                    break
            else:
                run_data["experiment"] = "default"
                
            data.append(run_data)
        
        if not data:
            console.print("[yellow]No finished runs found[/yellow]")
            return
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df = df.sort_values("val_acc", ascending=False)
        
        # Show summary by experiment
        console.print("\n[bold]Results Summary by Experiment:[/bold]")
        
        summary_table = Table(title="Experiment Performance")
        summary_table.add_column("Experiment", style="cyan")
        summary_table.add_column("Runs", style="green")
        summary_table.add_column("Best Val Acc", style="magenta")
        summary_table.add_column("Avg Val Acc", style="yellow")
        summary_table.add_column("Best Model", style="blue")
        
        for exp_name in df["experiment"].unique():
            exp_df = df[df["experiment"] == exp_name]
            best_run = exp_df.iloc[0]
            
            summary_table.add_row(
                exp_name,
                str(len(exp_df)),
                f"{best_run['val_acc']:.4f}",
                f"{exp_df['val_acc'].mean():.4f}",
                best_run['model']
            )
        
        console.print(summary_table)
        
        # Show top runs overall
        console.print("\n[bold]Top 10 Runs Overall:[/bold]")
        
        top_table = Table(title="Best Performing Runs")
        top_table.add_column("Rank", style="cyan")
        top_table.add_column("Run Name", style="green")
        top_table.add_column("Val Acc", style="magenta")
        top_table.add_column("Model", style="yellow")
        top_table.add_column("LR", style="blue")
        top_table.add_column("BS", style="blue")
        top_table.add_column("Loss", style="blue")
        
        for i, (_, run) in enumerate(df.head(10).iterrows()):
            top_table.add_row(
                str(i + 1),
                run['name'][:40] + "..." if len(run['name']) > 40 else run['name'],
                f"{run['val_acc']:.4f}",
                run['model'],
                f"{run['lr']:.1e}",
                str(run['batch_size']),
                run['loss']
            )
        
        console.print(top_table)
        
        # Export if requested
        if export:
            df.to_csv(export, index=False)
            console.print(f"\n[green]✓ Results exported to {export}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error fetching results: {e}[/red]")

def create_experiment(name, template=None):
    """Create a new experiment configuration"""
    exp_dir = "experiments"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Base template
    if template and os.path.exists(os.path.join(exp_dir, template)):
        with open(os.path.join(exp_dir, template), 'r') as f:
            config = json.load(f)
        console.print(f"[green]Using template: {template}[/green]")
    else:
        config = {
            "description": f"Experiment: {name}",
            "experiment_name": name,
            "parameters": {
                "model_name": ["efficientnet_b0"]
            },
            "fixed_parameters": {
                "batch_size": 32,
                "lr": 0.001,
                "loss": "ce",
                "max_epochs": 30,
                "early_stopping": 10,
                "img_size": 224,
                "num_workers": 4,
                "pretrained": True
            }
        }
    
    # Update name and description
    config["experiment_name"] = name
    config["description"] = f"Experiment: {name}"
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{timestamp}_{name}.json"
    filepath = os.path.join(exp_dir, filename)
    
    # Check if file exists
    if os.path.exists(filepath):
        if not Confirm.ask(f"File {filename} already exists. Overwrite?"):
            return
    
    # Save configuration
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    console.print(f"[green]✓ Created experiment configuration: {filepath}[/green]")
    console.print("\nEdit this file to customize your experiment parameters.")

def main():
    args = parse_args()
    
    if args.command == 'list':
        list_experiments()
    elif args.command == 'clean':
        clean_checkpoints(args.keep_best, args.dry_run)
    elif args.command == 'analyze':
        analyze_results(args.project, args.experiment, args.export)
    elif args.command == 'create':
        create_experiment(args.name, args.template)
    else:
        console.print("[yellow]No command specified. Use --help for usage.[/yellow]")

if __name__ == "__main__":
    main()