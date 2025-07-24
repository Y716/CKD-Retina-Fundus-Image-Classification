import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def classification_metrics(logits, targets):
    """
    Calculate classification metrics from logits and targets.
    
    Args:
        logits: Model output logits
        targets: Ground truth labels
        
    Returns:
        dict: Dictionary of metrics
    """
    # Convert logits to predictions
    preds = torch.argmax(logits, dim=1)
    
    # Calculate accuracy directly with torch
    accuracy = (preds == targets).float().mean()
    
    # For the rest of metrics, use sklearn but handle properly
    # Convert tensors to numpy arrays
    if isinstance(preds, torch.Tensor):
        preds_np = preds.detach().cpu().numpy()
    else:
        preds_np = np.array(preds)
        
    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
    else:
        targets_np = np.array(targets)
    
    # Handle empty arrays or single class
    if len(np.unique(targets_np)) <= 1:
        # If only one class, metrics are not well-defined
        return {"accuracy": accuracy}
    
    try:
        # Calculate metrics
        precision_macro = torch.tensor(precision_score(targets_np, preds_np, average='macro', zero_division=0))
        recall_macro = torch.tensor(recall_score(targets_np, preds_np, average='macro', zero_division=0))
        f1_macro = torch.tensor(f1_score(targets_np, preds_np, average='macro', zero_division=0))
        
        # Return metrics as a dictionary
        return {
            "accuracy": accuracy,
            "precision": precision_macro,
            "recall": recall_macro,
            "f1": f1_macro
        }
    except Exception as e:
        # Return only accuracy if other metrics fail
        print(f"Warning: Could not calculate all metrics: {e}")
        return {"accuracy": accuracy}


def confidence_metrics(logits, targets):
    """
    Calculate confidence-based metrics for classification.
    
    Args:
        logits: Model output logits
        targets: Ground truth labels
        
    Returns:
        dict: Dictionary of confidence metrics
    """
    # Get softmax probabilities
    probs = F.softmax(logits, dim=1)
    
    # Get max probability (confidence) for each prediction
    confidence, preds = torch.max(probs, dim=1)
    
    # Calculate mean confidence
    mean_confidence = confidence.mean()
    
    # Calculate mean confidence for correct and incorrect predictions
    correct_mask = preds == targets
    mean_confidence_correct = confidence[correct_mask].mean() if correct_mask.sum() > 0 else torch.tensor(0.0)
    
    incorrect_mask = ~correct_mask
    if incorrect_mask.sum() > 0:
        mean_confidence_incorrect = confidence[incorrect_mask].mean()
    else:
        mean_confidence_incorrect = torch.tensor(0.0)
    
    # Calculate calibration error (difference between confidence and accuracy)
    accuracy = correct_mask.float().mean()
    calibration_error = torch.abs(mean_confidence - accuracy)
    
    return {
        "mean_confidence": mean_confidence,
        "mean_confidence_correct": mean_confidence_correct,
        "mean_confidence_incorrect": mean_confidence_incorrect,
        "calibration_error": calibration_error
    }