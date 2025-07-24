#!/usr/bin/env python
"""
Script to run multiple experiments with different configurations.
This is useful for hyperparameter tuning and model selection.
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
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
import signal
import sys

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple fundus classification experiments")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to JSON config file with experiment parameters")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for this experiment group")
    parser.add_argument("--wandb_project", type=str, default="fundus-classification",
                        help="WandB project name")
    parser.add_argument("--sequential", action="store_true",
                        help="Run experiments sequentially instead of generating commands")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed output from each experiment")
    parser.add_argument("--dry_run", action="store_true",
                        help="Just print commands without executing them")
    parser.add_argument("--use_optimized", action="store_true",
                        help="Use the storage-optimized training script")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file with error handling"""
    console.print(f"[bold cyan]Loading configuration from {config_path}[/bold cyan]")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        if "parameters" not in config:
            raise ValueError("Config must contain a 'parameters' field")
            
        return config
    except FileNotFoundError:
        console.print(f"[bold red]Error: Config file {config_path} not found[/bold red]")
        sys.exit(1)
    except json.JSONDecodeError:
        console.print(f"[bold red]Error: Config file {config_path} is not valid JSON[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error loading config: {e}[/bold red]")
        sys.exit(1)

def generate_experiment_combinations(config):
    """Generate all combinations of parameters for experiments"""
    console.print("[bold cyan]Generating experiment combinations...[/bold cyan]")
    
    # Extract parameter values to sweep
    param_dict = {}
    for param, values in config["parameters"].items():
        param_dict[param] = values
    
    # Generate all combinations
    keys = param_dict.keys()
    values = param_dict.values()
    combinations = list(itertools.product(*values))
    
    # Convert to list of parameter dictionaries
    experiments = []
    for combination in combinations:
        experiment = dict(zip(keys, combination))
        
        # Add fixed parameters if they exist
        if "fixed_parameters" in config:
            for param, value in config["fixed_parameters"].items():
                experiment[param] = value
                
        experiments.append(experiment)
    
    console.print(f"[bold green]Generated {len(experiments)} experiment combinations[/bold green]")
    return experiments

def format_command(experiment, experiment_name, wandb_project, use_optimized=False):
    """Format a command to run an experiment with the given parameters"""
    # Choose which script to use
    script = "train_optimized.py" if use_optimized else "train.py"
    
    cmd = ["python", script]
    
    # Add experiment name as a tag
    if experiment_name:
        cmd.append(f"--tags={experiment_name}")
    
    # Add all parameters
    for param, value in experiment.items():
        cmd.append(f"--{param}={value}")
    
    # Add wandb project if specified
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        
    return cmd

def run_experiment(cmd, verbose=False):
    """Run a single experiment and capture its output"""
    start_time = time.time()
    
    try:
        if verbose:
            # Run with live output
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE if not verbose else None,
                stderr=subprocess.STDOUT if not verbose else None,
                text=True,
                bufsize=1
            )
            
            # Wait for process to complete
            return_code = process.wait()
            success = return_code == 0
            
        else:
            # Run with captured output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            success = result.returncode == 0
        
        elapsed = time.time() - start_time
        
        return {
            "success": success,
            "elapsed_time": elapsed
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }

def main():
    args = parse_args()
    
    # Setup signal handler for clean exit
    def signal_handler(sig, frame):
        console.print("\n[bold red]Experiment run interrupted. Exiting...[/bold red]")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate experiment name if not provided
    if not args.experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"exp_{timestamp}"
    
    # Generate all experiment combinations
    experiments = generate_experiment_combinations(config)
    
    # Print experiment plan
    console.print(f"[bold green]Running {len(experiments)} experiments for '{args.experiment_name}'[/bold green]")
    
    exp_table = Table(title="Experiment Configurations")
    
    # Add columns for all parameters
    for param in config["parameters"].keys():
        exp_table.add_column(param, style="cyan")
    
    # Add rows for each experiment
    for exp in experiments:
        exp_table.add_row(*[str(exp[param]) for param in config["parameters"].keys()])
    
    console.print(exp_table)
    
    # Generate commands for all experiments
    commands = [format_command(exp, args.experiment_name, args.wandb_project, args.use_optimized) for exp in experiments]
    
    # Display command examples
    if len(commands) > 0:
        console.print(f"[bold cyan]Example command:[/bold cyan] {' '.join(commands[0])}")
    
    # Ask for confirmation unless dry run
    if not args.dry_run and not args.sequential:
        response = input(f"\nGenerate {len(commands)} experiment commands? (y/n): ")
        if response.lower() != "y":
            console.print("[yellow]Operation cancelled.[/yellow]")
            return
    
    # Either run experiments or print commands
    if args.sequential and not args.dry_run:
        console.print("[bold green]Running experiments sequentially...[/bold green]")
        
        # Create a progress display
        layout = Layout()
        layout.split(
            Layout(name="progress"),
            Layout(name="current", size=3)
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            # Add main progress task
            task = progress.add_task("[cyan]Running experiments...", total=len(commands))
            
            # Initialize results tracking
            results = []
            
            # Run each experiment
            for i, cmd in enumerate(commands):
                # Update the current experiment info
                experiment_name = f"Experiment {i+1}/{len(commands)}"
                cmd_str = " ".join(cmd)
                
                # Add to layout
                progress.update(task, description=f"[cyan]Running {experiment_name}[/cyan]")
                
                # Display current command
                console.print(f"\n[bold]{'='*80}[/bold]")
                console.print(f"[bold cyan]{experiment_name}[/bold cyan]")
                console.print(f"[bold]Command:[/bold] {cmd_str}")
                console.print(f"[bold]{'='*80}[/bold]\n")
                
                # Run the experiment
                start_time = time.time()
                
                try:
                    if args.verbose:
                        # Run with visible output
                        console.print("[bold yellow]Experiment output:[/bold yellow]")
                        result = run_experiment(cmd, verbose=True)
                    else:
                        # Run with captured output
                        with console.status(f"[bold green]Running {experiment_name}...[/bold green]"):
                            result = run_experiment(cmd, verbose=False)
                    
                    # Record result
                    elapsed_time = result.get("elapsed_time", time.time() - start_time)
                    success = result.get("success", False)
                    
                    # Format and display result
                    if success:
                        console.print(f"[bold green]✓ {experiment_name} completed successfully in {elapsed_time:.1f} seconds[/bold green]")
                    else:
                        error = result.get("error", "Unknown error")
                        console.print(f"[bold red]✗ {experiment_name} failed after {elapsed_time:.1f} seconds: {error}[/bold red]")
                    
                    # Store result
                    results.append({
                        "experiment": i+1,
                        "command": cmd_str,
                        "success": success,
                        "elapsed_time": elapsed_time
                    })
                    
                except Exception as e:
                    console.print(f"[bold red]Error running experiment: {e}[/bold red]")
                    results.append({
                        "experiment": i+1,
                        "command": cmd_str,
                        "success": False,
                        "error": str(e),
                        "elapsed_time": time.time() - start_time
                    })
                
                # Update progress
                progress.update(task, advance=1)
                
            # Display summary
            console.print("\n[bold green]Experiments completed![/bold green]")
            
            # Create results table
            results_table = Table(title="Experiment Results")
            results_table.add_column("Experiment", style="cyan")
            results_table.add_column("Status", style="green")
            results_table.add_column("Time (s)", style="blue")
            
            successful = 0
            for result in results:
                status = "[green]✓ Success[/green]" if result["success"] else "[red]✗ Failed[/red]"
                results_table.add_row(
                    f"{result['experiment']}/{len(commands)}",
                    status,
                    f"{result['elapsed_time']:.1f}"
                )
                if result["success"]:
                    successful += 1
            
            console.print(results_table)
            console.print(f"[bold]Summary:[/bold] {successful}/{len(commands)} experiments completed successfully")
            
    else:
        # Just print the commands so they can be run manually or in parallel
        console.print("[bold yellow]Commands to run:[/bold yellow]")
        
        commands_table = Table(title="Experiment Commands")
        commands_table.add_column("Experiment", style="cyan")
        commands_table.add_column("Command", style="green")
        
        for i, cmd in enumerate(commands):
            commands_table.add_row(f"{i+1}", " ".join(cmd))
        
        console.print(commands_table)
        
        if args.dry_run:
            console.print("[bold yellow]This was a dry run. No commands were executed.[/bold yellow]")
        else:
            console.print("[bold yellow]Copy and paste these commands to run them individually.[/bold yellow]")

if __name__ == "__main__":
    main()