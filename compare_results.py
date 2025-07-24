#!/usr/bin/env python
"""
Script to compare results from multiple experiments using WandB API
"""

import argparse
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
import os

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Compare experiment results from WandB")
    parser.add_argument("--project", type=str, default="fundus-classification",
                        help="WandB project name")
    parser.add_argument("--entity", type=str, default=None,
                        help="WandB entity (username or team name)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Filter runs by tag")
    parser.add_argument("--metric", type=str, default="val_acc",
                        help="Metric to compare (default: val_acc)")
    parser.add_argument("--group-by", type=str, default="model_name",
                        help="Parameter to group results by")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--best-n", type=int, default=5,
                        help="Number of best runs to display")
    return parser.parse_args()

def fetch_runs(api, project, entity=None, tag=None):
    """Fetch runs from WandB"""
    filters = {}
    if tag:
        filters["tags"] = tag
        
    runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)
    console.print(f"[bold green]Found {len(runs)} runs in project {project}{f' with tag {tag}' if tag else ''}[/bold green]")
    return runs

def extract_run_data(runs, metric):
    """Extract relevant data from runs"""
    run_data = []
    
    for run in runs:
        # Get run configuration
        config = run.config
        
        # Skip runs that don't have the metric we're looking for
        if metric not in run.summary:
            console.print(f"[yellow]Warning: Run {run.name} does not have metric {metric}, skipping...[/yellow]")
            continue
            
        # Extract metric value
        metric_value = run.summary[metric]
        
        # Extract common parameters of interest
        data = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            metric: metric_value
        }
        
        # Add configuration parameters
        for key, value in config.items():
            if isinstance(value, dict) or isinstance(value, list):
                continue  # Skip nested configurations
            data[key] = value
            
        run_data.append(data)
        
    return pd.DataFrame(run_data)

def rank_runs(df, metric, ascending=False):
    """Rank runs by the specified metric"""
    # Only consider finished runs
    df_finished = df[df['state'] == 'finished'].copy()
    
    if df_finished.empty:
        console.print("[yellow]Warning: No finished runs found.[/yellow]")
        return df
        
    # Sort by the metric
    df_sorted = df_finished.sort_values(by=metric, ascending=ascending)
    
    # Add rank column
    df_sorted['rank'] = range(1, len(df_sorted) + 1)
    
    return df_sorted

def group_and_compare(df, metric, group_by):
    """Group runs by parameter and compare metric values"""
    if df.empty:
        console.print("[yellow]No data to group and compare.[/yellow]")
        return
        
    # Group by the specified parameter
    grouped = df.groupby(group_by)[metric].agg(['mean', 'std', 'max', 'min', 'count'])
    grouped = grouped.sort_values(by='mean', ascending=False)
    
    return grouped

def visualize_results(df, metric, group_by, output_dir):
    """Create visualizations of results"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if df.empty:
        console.print("[yellow]No data to visualize.[/yellow]")
        return
        
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 1. Create boxplot grouped by the parameter
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x=group_by, y=metric, data=df)
    ax.set_title(f"{metric} by {group_by}", fontsize=16)
    ax.set_xlabel(group_by, fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_by_{group_by}_boxplot.png"))
    console.print(f"[green]Saved boxplot to {output_dir}/{metric}_by_{group_by}_boxplot.png[/green]")
    
    # 2. Create barplot with error bars for mean performance
    plt.figure(figsize=(14, 8))
    grouped = df.groupby(group_by)[metric].agg(['mean', 'std']).reset_index()
    grouped = grouped.sort_values(by='mean', ascending=False)
    
    ax = sns.barplot(x=group_by, y='mean', data=grouped, yerr=grouped['std'])
    ax.set_title(f"Mean {metric} by {group_by}", fontsize=16)
    ax.set_xlabel(group_by, fontsize=14)
    ax.set_ylabel(f"Mean {metric}", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_by_{group_by}_barplot.png"))
    console.print(f"[green]Saved barplot to {output_dir}/{metric}_by_{group_by}_barplot.png[/green]")
    
    # 3. Create heatmap for parameter combinations if we have enough data
    if len(df) >= 4 and len(df[group_by].unique()) >= 2:
        # Find another parameter to use for the heatmap
        possible_params = [col for col in df.columns if col not in [metric, group_by, 'run_id', 'run_name', 'state', 'rank'] 
                          and len(df[col].unique()) >= 2]
        
        if possible_params:
            second_param = possible_params[0]  # Use the first available parameter
            
            plt.figure(figsize=(12, 10))
            pivot = df.pivot_table(index=group_by, columns=second_param, values=metric, aggfunc='mean')
            ax = sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f")
            ax.set_title(f"Mean {metric} by {group_by} and {second_param}", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{metric}_heatmap.png"))
            console.print(f"[green]Saved heatmap to {output_dir}/{metric}_heatmap.png[/green]")

def display_best_runs(df, metric, best_n=5):
    """Display table of best runs"""
    if df.empty:
        console.print("[yellow]No runs to display.[/yellow]")
        return
        
    # Get the best n runs
    best_runs = df.head(best_n)
    
    # Create a table
    table = Table(title=f"Top {best_n} Runs by {metric}")
    
    # Add columns
    table.add_column("Rank", style="cyan")
    table.add_column("Run Name", style="green")
    table.add_column(metric, style="magenta")
    
    # Add parameters of interest
    params_of_interest = [col for col in df.columns if col not in ['run_id', 'run_name', 'state', 'rank', metric]]
    for param in params_of_interest:
        table.add_column(param)
    
    # Add rows
    for _, run in best_runs.iterrows():
        row = [
            str(run['rank']),
            run['run_name'],
            f"{run[metric]:.4f}"
        ]
        
        # Add parameter values
        for param in params_of_interest:
            if param in run:
                row.append(str(run[param]))
            else:
                row.append("-")
                
        table.add_row(*row)
    
    # Print the table
    console.print(table)

def main():
    args = parse_args()
    
    # Initialize WandB API
    api = wandb.Api()
    
    # Fetch runs
    console.print(f"[bold]Fetching runs from WandB project {args.project}...[/bold]")
    runs = fetch_runs(api, args.project, args.entity, args.tag)
    
    # Extract data
    console.print(f"[bold]Extracting run data for metric '{args.metric}'...[/bold]")
    df = extract_run_data(runs, args.metric)
    
    if df.empty:
        console.print("[red]No valid runs found with the specified metric.[/red]")
        return
    
    # Rank runs
    df_ranked = rank_runs(df, args.metric, ascending=False if 'acc' in args.metric or 'f1' in args.metric else True)
    
    # Group and compare
    console.print(f"[bold]Comparing runs grouped by '{args.group_by}'...[/bold]")
    grouped = group_and_compare(df_ranked, args.metric, args.group_by)
    
    if grouped is not None:
        # Display grouped results
        console.print("[bold cyan]Performance by Group:[/bold cyan]")
        console.print(grouped)
    
    # Display best runs
    console.print(f"[bold cyan]Top {args.best_n} Runs:[/bold cyan]")
    display_best_runs(df_ranked, args.metric, args.best_n)
    
    # Visualize results
    console.print("[bold]Creating visualizations...[/bold]")
    visualize_results(df_ranked, args.metric, args.group_by, args.output_dir)
    
    console.print("[bold green]Analysis complete![/bold green]")

if __name__ == "__main__":
    main()