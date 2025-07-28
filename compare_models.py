#!/usr/bin/env python
"""
Script to compare models by key metrics (train_loss, train_acc, val_loss, val_acc)
and create visualizations for easy comparison.
"""

import argparse
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
import os
import numpy as np

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Compare models by key metrics")
    parser.add_argument("--project", type=str, default="fundus-classification",
                        help="WandB project name")
    parser.add_argument("--entity", type=str, default=None,
                        help="WandB entity (username or team name)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Filter runs by tag")
    parser.add_argument("--metric", type=str, default="val_acc",
                        help="Primary metric to compare (default: val_acc)")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                        help="Directory to save comparison results")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of top models to show")
    return parser.parse_args()

def fetch_runs(api, project, entity=None, tag=None):
    """Fetch runs from WandB"""
    filters = {}
    if tag:
        filters["tags"] = tag
        
    runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)
    console.print(f"[bold green]Found {len(runs)} runs in project {project}{f' with tag {tag}' if tag else ''}[/bold green]")
    return runs

def extract_key_metrics(runs):
    """Extract key metrics from runs"""
    metrics_data = []
    
    for run in runs:
        # Get run configuration
        config = run.config
        
        # Basic run info
        run_info = {
            "run_id": run.id,
            "run_name": run.name,
            "model": config.get("model_name", "unknown"),
            "state": run.state
        }
        
        # Extract key metrics if available
        for metric in ["val_acc", "val_loss", "train_acc", "train_loss"]:
            if metric in run.summary:
                run_info[metric] = run.summary[metric]
            else:
                run_info[metric] = None
        
        # Add other useful configuration parameters
        for param in ["batch_size", "lr", "img_size", "loss", "max_epochs"]:
            if param in config:
                run_info[param] = config[param]
        
        metrics_data.append(run_info)
    
    return pd.DataFrame(metrics_data)

def create_comparison_table(df, metric, ascending=False, top_n=10):
    """Create a table comparing top models by the specified metric"""
    # Filter to only finished runs
    df_finished = df[df['state'] == 'finished'].copy()
    
    if df_finished.empty:
        console.print("[yellow]Warning: No finished runs found.[/yellow]")
        return None
    
    # Sort by the primary metric
    if metric in df_finished.columns:
        df_sorted = df_finished.sort_values(by=metric, ascending=ascending)
        df_sorted = df_sorted.head(top_n)
        
        # Create comparison table
        table = Table(title=f"Top {top_n} Models by {metric}")
        
        # Add columns
        table.add_column("Rank", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Val Acc", style="magenta")
        table.add_column("Val Loss", style="magenta")
        table.add_column("Train Acc", style="blue")
        table.add_column("Train Loss", style="blue")
        table.add_column("Learning Rate", style="yellow")
        table.add_column("Batch Size", style="yellow")
        
        # Add rows
        for i, (_, run) in enumerate(df_sorted.iterrows()):
            table.add_row(
                str(i+1),
                f"{run['model']}",
                f"{run['val_acc']:.4f}" if pd.notnull(run['val_acc']) else "N/A",
                f"{run['val_loss']:.4f}" if pd.notnull(run['val_loss']) else "N/A",
                f"{run['train_acc']:.4f}" if pd.notnull(run['train_acc']) else "N/A",
                f"{run['train_loss']:.4f}" if pd.notnull(run['train_loss']) else "N/A",
                f"{run['lr']}" if pd.notnull(run.get('lr')) else "N/A",
                f"{run['batch_size']}" if pd.notnull(run.get('batch_size')) else "N/A"
            )
        
        return table, df_sorted
    else:
        console.print(f"[yellow]Warning: Metric '{metric}' not found in runs.[/yellow]")
        return None, None

def create_comparison_plots(df, output_dir):
    """Create plots comparing models on key metrics"""
    if df is None or df.empty:
        console.print("[yellow]No data available for plotting.[/yellow]")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 1. Val Accuracy comparison
    if 'val_acc' in df.columns and df['val_acc'].notna().any():
        plt.figure(figsize=(12, 6))
        # Sort by val_acc for the plot
        plot_df = df.sort_values('val_acc', ascending=False).reset_index(drop=True)
        sns.barplot(x='model', y='val_acc', data=plot_df)
        plt.title('Validation Accuracy by Model', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Validation Accuracy', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'val_acc_comparison.png'))
        plt.close()
    
    # 2. Val Loss comparison
    if 'val_loss' in df.columns and df['val_loss'].notna().any():
        plt.figure(figsize=(12, 6))
        # Sort by val_loss for the plot
        plot_df = df.sort_values('val_loss', ascending=True).reset_index(drop=True)
        sns.barplot(x='model', y='val_loss', data=plot_df)
        plt.title('Validation Loss by Model', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Validation Loss', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'val_loss_comparison.png'))
        plt.close()
    
    # 3. Scatter plot of val_acc vs. val_loss
    if 'val_acc' in df.columns and 'val_loss' in df.columns and df['val_acc'].notna().any() and df['val_loss'].notna().any():
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='val_loss', y='val_acc', hue='model', s=100, data=df)
        plt.title('Validation Accuracy vs. Loss', fontsize=16)
        plt.xlabel('Validation Loss', fontsize=14)
        plt.ylabel('Validation Accuracy', fontsize=14)
        # Add labels for each point
        for i, row in df.iterrows():
            plt.text(row['val_loss'], row['val_acc'], row['model'], 
                    fontsize=9, ha='right', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'val_acc_vs_loss.png'))
        plt.close()
    
    # 4. Grouped bar chart comparing train/val metrics
    metrics_to_plot = []
    if 'val_acc' in df.columns and 'train_acc' in df.columns and df['val_acc'].notna().any() and df['train_acc'].notna().any():
        metrics_to_plot.append(('acc', 'Accuracy'))
    if 'val_loss' in df.columns and 'train_loss' in df.columns and df['val_loss'].notna().any() and df['train_loss'].notna().any():
        metrics_to_plot.append(('loss', 'Loss'))
    
    for metric_suffix, metric_name in metrics_to_plot:
        plt.figure(figsize=(14, 6))
        
        # Prepare data for grouped bar chart
        models = df['model'].unique()
        train_vals = [df[df['model'] == model][f'train_{metric_suffix}'].values[0] if not df[df['model'] == model][f'train_{metric_suffix}'].isnull().all() else np.nan for model in models]
        val_vals = [df[df['model'] == model][f'val_{metric_suffix}'].values[0] if not df[df['model'] == model][f'val_{metric_suffix}'].isnull().all() else np.nan for model in models]
        
        # Set up bar positions
        x = np.arange(len(models))
        width = 0.35
        
        # Create the grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 6))
        rects1 = ax.bar(x - width/2, train_vals, width, label=f'Train {metric_name}')
        rects2 = ax.bar(x + width/2, val_vals, width, label=f'Val {metric_name}')
        
        # Add labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_name)
        ax.set_title(f'Training and Validation {metric_name} by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if not np.isnan(height):
                    ax.annotate(f'{height:.4f}',
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric_suffix}_train_val_comparison.png'))
        plt.close()
    
    console.print(f"[green]Saved comparison plots to {output_dir}[/green]")

def export_results_to_csv(df, output_dir):
    """Export comparison results to CSV"""
    if df is None or df.empty:
        console.print("[yellow]No data available for export.[/yellow]")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the DataFrame to CSV
    output_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(output_path, index=False)
    console.print(f"[green]Exported comparison results to {output_path}[/green]")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize WandB API
    api = wandb.Api()
    
    # Fetch runs
    console.print(f"[bold]Fetching runs from WandB project {args.project}...[/bold]")
    runs = fetch_runs(api, args.project, args.entity, args.tag)
    
    # Extract key metrics
    console.print(f"[bold]Extracting key metrics from runs...[/bold]")
    df = extract_key_metrics(runs)
    
    if df.empty:
        console.print("[red]No valid runs found with metrics.[/red]")
        return
    
    # Determine if the metric should be sorted in ascending or descending order
    ascending = "loss" in args.metric  # True for loss metrics, False for accuracy metrics
    
    # Create comparison table
    console.print(f"[bold]Creating comparison table for top {args.top_n} models by {args.metric}...[/bold]")
    table_result = create_comparison_table(df, args.metric, ascending, args.top_n)
    
    if table_result:
        table, top_df = table_result
        console.print(table)
        
        # Create comparison plots
        console.print("[bold]Creating comparison plots...[/bold]")
        create_comparison_plots(top_df, args.output_dir)
        
        # Export results to CSV
        export_results_to_csv(top_df, args.output_dir)
    else:
        console.print("[red]Could not create comparison table.[/red]")
    
    console.print("[bold green]Comparison complete![/bold green]")

if __name__ == "__main__":
    main()