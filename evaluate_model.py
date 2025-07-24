#!/usr/bin/env python
"""
Script to evaluate a trained model on a test set and generate detailed reports
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from sklearn.metrics import classification_report, confusion_matrix
import lightning.pytorch as pl
from model.classifier.model import ClassificationModel
from model.classifier.dataset import FundusDataModule
import wandb

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained fundus classification model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test images")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--log_wandb", action="store_true",
                        help="Log results to WandB")
    parser.add_argument("--visualize_samples", type=int, default=10,
                        help="Number of samples to visualize per class")
    return parser.parse_args()

def load_model(checkpoint_path):
    """Load a trained model from checkpoint"""
    console.print(f"[bold cyan]Loading model from {checkpoint_path}[/bold cyan]")
    
    try:
        model = ClassificationModel.load_from_checkpoint(checkpoint_path)
        model.eval()
        model.freeze()
        return model
    except Exception as e:
        console.print(f"[bold red]Error loading model: {e}[/bold red]")
        raise

def prepare_data(test_dir, batch_size, num_workers, img_size=224):
    """Prepare test data loader"""
    console.print(f"[bold cyan]Preparing test data from {test_dir}[/bold cyan]")
    
    # Create a simple data module
    data_module = FundusDataModule(
        train_dir=test_dir,  # This won't be used for evaluation
        val_dir=test_dir,    # We'll use this as our test set
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers
    )
    
    # Set up the data module
    data_module.setup()
    
    # Return the test dataloader
    return data_module.val_dataloader(), data_module.val_dataset

def evaluate_model(model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Evaluate model on test data"""
    console.print(f"[bold cyan]Evaluating model on {device}...[/bold cyan]")
    
    model = model.to(device)
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        with Progress() as progress:
            task = progress.add_task("[green]Evaluating...", total=len(dataloader))
            
            for batch in dataloader:
                images, labels = batch
                images = images.to(device)
                
                # Forward pass
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                progress.update(task, advance=1)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def generate_metrics_report(preds, labels, class_names):
    """Generate and display classification metrics"""
    console.print("[bold cyan]Generating classification report...[/bold cyan]")
    
    # Calculate metrics
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Display metrics table
    table = Table(title="Classification Report")
    
    # Add columns
    table.add_column("Class", style="cyan")
    table.add_column("Precision", style="green")
    table.add_column("Recall", style="green")
    table.add_column("F1-Score", style="green")
    table.add_column("Support", style="green")
    
    # Add rows for each class
    for i, class_name in enumerate(class_names):
        if class_name in report_df.index:
            row = report_df.loc[class_name]
            table.add_row(
                class_name,
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1-score']:.4f}",
                f"{row['support']}"
            )
    
    # Add aggregate rows
    for agg in ['macro avg', 'weighted avg']:
        if agg in report_df.index:
            row = report_df.loc[agg]
            table.add_row(
                agg,
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1-score']:.4f}",
                f"{row['support']}"
            )
    
    # Calculate accuracy
    accuracy = (preds == labels).mean()
    table.add_row(
        "Accuracy",
        f"{accuracy:.4f}",
        "",
        "",
        f"{len(labels)}"
    )
    
    console.print(table)
    
    return report_df, accuracy

def plot_confusion_matrix(preds, labels, class_names, output_dir):
    """Plot and save confusion matrix"""
    console.print("[bold cyan]Generating confusion matrix...[/bold cyan]")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'))
    
    console.print(f"[green]Confusion matrices saved to {output_dir}[/green]")

def plot_sample_predictions(dataset, preds, labels, probs, class_names, samples_per_class, output_dir):
    """Plot sample predictions (correct and incorrect)"""
    console.print("[bold cyan]Generating sample predictions visualization...[/bold cyan]")
    
    # Create directory if it doesn't exist
    samples_dir = os.path.join(output_dir, 'sample_predictions')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Get file paths from dataset
    file_paths = [dataset.samples[i][0] for i in range(len(dataset))]
    
    # Map to store indices of samples for each class
    class_indices = {
        cls: {
            'correct': [],
            'incorrect': []
        } for cls in range(len(class_names))
    }
    
    # Collect indices
    for i, (true, pred) in enumerate(zip(labels, preds)):
        if true == pred:
            class_indices[true]['correct'].append(i)
        else:
            class_indices[true]['incorrect'].append(i)
    
    # Plot samples for each class
    for cls in range(len(class_names)):
        correct_samples = class_indices[cls]['correct']
        incorrect_samples = class_indices[cls]['incorrect']
        
        # Plot correct predictions
        if correct_samples:
            n_samples = min(samples_per_class, len(correct_samples))
            sample_indices = np.random.choice(correct_samples, size=n_samples, replace=False)
            
            fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 4, 4))
            if n_samples == 1:
                axes = [axes]
                
            for i, idx in enumerate(sample_indices):
                img = Image.open(file_paths[idx])
                axes[i].imshow(img)
                prob = probs[idx][preds[idx]] * 100
                axes[i].set_title(f"True: {class_names[labels[idx]]}\nPred: {class_names[preds[idx]]}\nConf: {prob:.1f}%")
                axes[i].axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(samples_dir, f'class_{cls}_correct.png'))
            plt.close()
        
        # Plot incorrect predictions
        if incorrect_samples:
            n_samples = min(samples_per_class, len(incorrect_samples))
            sample_indices = np.random.choice(incorrect_samples, size=n_samples, replace=False)
            
            fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 4, 4))
            if n_samples == 1:
                axes = [axes]
                
            for i, idx in enumerate(sample_indices):
                img = Image.open(file_paths[idx])
                axes[i].imshow(img)
                prob = probs[idx][preds[idx]] * 100
                axes[i].set_title(f"True: {class_names[labels[idx]]}\nPred: {class_names[preds[idx]]}\nConf: {prob:.1f}%")
                axes[i].axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(samples_dir, f'class_{cls}_incorrect.png'))
            plt.close()
    
    console.print(f"[green]Sample predictions saved to {samples_dir}[/green]")

def plot_class_distribution(labels, class_names, output_dir):
    """Plot class distribution in test set"""
    console.print("[bold cyan]Generating class distribution plot...[/bold cyan]")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Count occurrences of each class
    class_counts = pd.Series(labels).value_counts().sort_index()
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=list(range(len(class_counts))), y=class_counts.values)
    
    # Add count labels on top of bars
    for i, count in enumerate(class_counts.values):
        ax.text(i, count + 0.1, str(count), ha='center')
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in Test Set')
    plt.xticks(list(range(len(class_names))), class_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    
    console.print(f"[green]Class distribution plot saved to {output_dir}[/green]")

def log_to_wandb(preds, labels, probs, class_names, report_df, accuracy, output_dir):
    """Log evaluation results to WandB"""
    console.print("[bold cyan]Logging results to WandB...[/bold cyan]")
    
    # Initialize a new run
    wandb.init(project="fundus-classification-evaluation", name="model-evaluation")
    
    # Log metrics
    wandb.log({
        "accuracy": accuracy,
        "precision_macro": report_df.loc['macro avg', 'precision'],
        "recall_macro": report_df.loc['macro avg', 'recall'],
        "f1_macro": report_df.loc['macro avg', 'f1-score']
    })
    
    # Log confusion matrix
    cm = confusion_matrix(labels, preds)
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=probs,
            y_true=labels,
            preds=preds,
            class_names=class_names
        )
    })
    
    # Log class metrics
    class_metrics = {}
    for i, cls in enumerate(class_names):
        if cls in report_df.index:
            class_metrics[f"precision_{cls}"] = report_df.loc[cls, 'precision']
            class_metrics[f"recall_{cls}"] = report_df.loc[cls, 'recall']
            class_metrics[f"f1_{cls}"] = report_df.loc[cls, 'f1-score']
    
    wandb.log(class_metrics)
    
    # Log images
    wandb.log({
        "confusion_matrix_plot": wandb.Image(os.path.join(output_dir, 'confusion_matrix.png')),
        "class_distribution": wandb.Image(os.path.join(output_dir, 'class_distribution.png'))
    })
    
    # Finish the run
    wandb.finish()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Prepare data
    dataloader, dataset = prepare_data(
        args.test_dir,
        args.batch_size,
        args.num_workers,
        img_size=model.hparams.get('img_size', 224)
    )
    
    # Get class names
    class_names = dataset.classes
    
    # Evaluate model
    preds, labels, probs = evaluate_model(model, dataloader)
    
    # Generate metrics report
    report_df, accuracy = generate_metrics_report(preds, labels, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(preds, labels, class_names, args.output_dir)
    
    # Plot class distribution
    plot_class_distribution(labels, class_names, args.output_dir)
    
    # Plot sample predictions
    plot_sample_predictions(dataset, preds, labels, probs, class_names, args.visualize_samples, args.output_dir)
    
    # Log to WandB if requested
    if args.log_wandb:
        log_to_wandb(preds, labels, probs, class_names, report_df, accuracy, args.output_dir)
    
    # Save results to CSV
    report_df.to_csv(os.path.join(args.output_dir, 'classification_report.csv'))
    
    console.print(f"[bold green]Evaluation complete! Results saved to {args.output_dir}[/bold green]")

if __name__ == "__main__":
    main()