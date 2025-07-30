#!/usr/bin/env python
"""
Script to run multiple experiments with different configurations.
Automatically handles experiment naming and parameter combinations.
"""

import subprocess
import argparse
import os
import json
import itertools
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import signal
import sys

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple fundus classification experiments")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to JSON config file with experiment parameters")
    parser.add_argument("--sequential", action="store_true", default=True,
                        help="Run experiments sequentially (default: True)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Just print commands without executing them")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device to use")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file"""
    console.print(f"[bold cyan]Loading configuration from {config_path}[/bold cyan]")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if "parameters" not in config:
            raise ValueError("Config must contain a 'parameters' field")
            
        return config
    except Exception as e:
        console.print(f"[bold red]Error loading config: {e}[/bold red]")
        sys.exit(1)

def generate_experiment_combinations(config):
    """Generate all combinations of parameters for experiments"""
    console.print("[bold cyan]Generating experiment combinations...[/bold cyan]")
    
    # Extract parameter values to sweep
    param_dict = config["parameters"]
    
    # Handle special case where focal_gamma should only be used with focal loss
    if "loss" in param_dict and "focal_gamma" in param_dict:
        combinations = []
        focal_gammas = param_dict.pop("focal_gamma")
        
        # Generate base combinations
        keys = list(param_dict.keys())
        values = list(param_dict.values())
        base_combinations = list(itertools.product(*values))
        
        # Add focal_gamma only for focal loss
        for combo in base_combinations:
            combo_dict = dict(zip(keys, combo))
            if combo_dict.get("loss") == "focal":
                for gamma in focal_gammas:
                    full_combo = combo_dict.copy()
                    full_combo["focal_gamma"] = gamma
                    combinations.append(full_combo)
            else:
                combinations.append(combo_dict)
    else:
        # Normal case: all combinations
        keys = list(param_dict.keys())
        values = list(param_dict.values())
        combos = list(itertools.product(*values))
        combinations = [dict(zip(keys, combo)) for combo in combos]
    
    # Add fixed parameters to each combination
    experiments = []
    for combo in combinations:
        experiment = combo.copy()
        if "fixed_parameters" in config:
            experiment.update(config["fixed_parameters"])
        
        # Add experiment name if provided in config
        if "experiment_name" in config:
            experiment["experiment_name"] = config["experiment_name"]
            
        experiments.append(experiment)
    
    console.print(f"[bold green]Generated {len(experiments)} experiment combinations[/bold green]")
    return experiments

def format_command(experiment, gpu=0):
    """Format a command to run an experiment with the given parameters"""
    cmd = ["python", "train.py"]
    
    # Set GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    # Add all parameters
    for param, value in experiment.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{param}")
            else:
                cmd.append(f"--{param}={value}")
    
    return cmd, env

def run_experiment(cmd, env, experiment_desc):
    """Run a single experiment"""
    start_time = time.time()
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Capture key metrics from output
        last_metrics = {}
        output_lines = []
        
        for line in process.stdout:
            # Store last 50 lines for error diagnosis
            output_lines.append(line.strip())
            if len(output_lines) > 50:
                output_lines.pop(0)
                
            # Print line in real-time
            print(line.strip())
            
            # Look for metric lines
            if "train_loss:" in line or "val_acc:" in line:
                # Parse metrics from the line
                parts = line.strip().split("|")
                for part in parts:
                    if ":" in part:
                        try:
                            # Split only on first colon to handle time formats
                            key_value = part.split(":", 1)
                            if len(key_value) == 2:
                                key = key_value[0].strip()
                                value = key_value[1].strip()
                                # Try to convert to float
                                try:
                                    value = float(value)
                                    last_metrics[key] = value
                                except ValueError:
                                    # Not a number, skip
                                    pass
                        except Exception as e:
                            # Skip problematic parts
                            pass
        
        return_code = process.wait()
        success = return_code == 0
        elapsed = time.time() - start_time
        
        # If failed, include last output lines for debugging
        error_msg = None
        if not success:
            error_msg = f"Process exited with code {return_code}. Last output:\n" + "\n".join(output_lines[-10:])
        
        return {
            "success": success,
            "elapsed_time": elapsed,
            "metrics": last_metrics,
            "error": error_msg,
            "return_code": return_code
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
            "metrics": {},
            "return_code": -1
        }

def main():
    args = parse_args()
    
    # Setup signal handler
    def signal_handler(sig, frame):
        console.print("\n[bold red]Experiment run interrupted. Exiting...[/bold red]")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate experiments
    experiments = generate_experiment_combinations(config)
    
    # Display experiment plan
    exp_name = config.get("experiment_name", "experiment")
    console.print(f"\n[bold green]Running {len(experiments)} experiments for '{exp_name}'[/bold green]\n")
    
    # Show experiment table
    if len(experiments) <= 20:  # Only show table for reasonable number of experiments
        table = Table(title="Planned Experiments")
        
        # Add columns for varying parameters only
        varying_params = list(config["parameters"].keys())
        for param in varying_params:
            table.add_column(param, style="cyan")
        
        # Add rows
        for exp in experiments:
            row = [str(exp.get(param, "")) for param in varying_params]
            table.add_row(*row)
        
        console.print(table)
    
    # Ask for confirmation
    if not args.dry_run:
        response = input(f"\nRun {len(experiments)} experiments? (y/n): ")
        if response.lower() != "y":
            console.print("[yellow]Operation cancelled.[/yellow]")
            return
    
    # Generate commands
    commands_and_envs = [format_command(exp, args.gpu) for exp in experiments]
    
    if args.dry_run:
        console.print("\n[bold yellow]Dry run - commands that would be executed:[/bold yellow]\n")
        for i, (cmd, _) in enumerate(commands_and_envs):
            console.print(f"[{i+1}] {' '.join(cmd)}")
        return
    
    # Run experiments
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Running experiments...", total=len(experiments))
        
        for i, (exp, (cmd, env)) in enumerate(zip(experiments, commands_and_envs)):
            # Create experiment description
            varying_values = [f"{k}={exp.get(k)}" for k in config["parameters"].keys() if k in exp]
            exp_desc = f"Exp {i+1}/{len(experiments)}: {', '.join(varying_values)}"
            
            progress.update(task, description=f"[cyan]{exp_desc}[/cyan]")
            console.print(f"\n[bold]{'='*80}[/bold]")
            console.print(f"[bold cyan]{exp_desc}[/bold cyan]")
            console.print(f"[bold]{'='*80}[/bold]\n")
            
            # Run experiment
            result = run_experiment(cmd, env, exp_desc)
            result["experiment"] = exp_desc
            results.append(result)
            
            # Show result
            if result["success"]:
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in result["metrics"].items()])
                console.print(f"[green]✓ Completed in {result['elapsed_time']:.1f}s - {metrics_str}[/green]")
            else:
                console.print(f"[red]✗ Failed after {result['elapsed_time']:.1f}s[/red]")
                if result.get("error"):
                    console.print(f"[red]Error: {result['error']}[/red]")
            
            progress.update(task, advance=1)
    
    # Summary
    console.print("\n[bold green]Experiment Summary[/bold green]\n")
    
    summary_table = Table(title="Results Summary")
    summary_table.add_column("Experiment", style="cyan")
    summary_table.add_column("Status", style="green")
    summary_table.add_column("Time", style="blue")
    summary_table.add_column("Best Val Acc", style="magenta")
    
    successful = 0
    best_val_acc = 0
    best_exp = None
    
    for result in results:
        status = "[green]✓[/green]" if result["success"] else "[red]✗[/red]"
        val_acc = result["metrics"].get("val_acc", 0)
        
        summary_table.add_row(
            result["experiment"],
            status,
            f"{result['elapsed_time']:.1f}s",
            f"{val_acc:.4f}" if val_acc > 0 else "-"
        )
        
        if result["success"]:
            successful += 1
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_exp = result["experiment"]
    
    console.print(summary_table)
    console.print(f"\n[bold]Success rate: {successful}/{len(experiments)} ({100*successful/len(experiments):.1f}%)[/bold]")
    
    if best_exp:
        console.print(f"[bold green]Best validation accuracy: {best_val_acc:.4f} ({best_exp})[/bold green]")

if __name__ == "__main__":
    main()