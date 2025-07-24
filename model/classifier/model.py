import lightning.pytorch as pl
import torch
import torch.nn as nn
import timm
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import confusion_matrix


class ClassificationModel(pl.LightningModule):
    def __init__(self, 
                 model_name, 
                 num_classes,
                 in_channels=3,
                 loss_fn=None,
                 metric_fn=None,
                 lr=1e-3,
                 weight_decay=1e-4,
                 scheduler_type=None,
                 scheduler_params=None):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn', 'metric_fn'])
        
        # Initialize model from timm
        self.model = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=num_classes, 
            in_chans=in_channels
        )
        
        # Set loss function or default to CrossEntropyLoss
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        
        # Custom metric function
        self.metric_fn = metric_fn
        
        # Training hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}
        
        # Initialize metrics for tracking
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        
        # For confusion matrix and predictions tracking
        self.val_targets = []
        self.val_preds = []
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Calculate and log accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # Log additional metrics if provided
        if self.metric_fn:
            metrics = self.metric_fn(logits, y)
            # Prefix with "train_" and log each metric individually to avoid conflicts
            for k, v in metrics.items():
                if k != "accuracy":  # Avoid duplicate with train_acc above
                    self.log(f"train_{k}", v, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Calculate predictions
        preds = torch.argmax(logits, dim=1)
        
        # Store targets and predictions for confusion matrix
        self.val_targets.append(y.cpu())
        self.val_preds.append(preds.cpu())
        
        # Update metrics
        acc = self.val_acc(preds, y)
        f1 = self.val_f1(preds, y)
        
        # Log basic metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_f1", f1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log additional metrics if provided, individually to avoid conflicts
        if self.metric_fn:
            metrics = self.metric_fn(logits, y)
            # Log each metric individually, avoiding duplicates with metrics already logged
            for k, v in metrics.items():
                if k not in ["accuracy", "f1"]:  # Avoid duplicates
                    self.log(f"val_{k}", v, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return {"val_loss": loss, "val_preds": preds, "val_targets": y}

    def on_validation_epoch_end(self):
        """Calculate and log confusion matrix at the end of validation epoch"""
        if not self.val_targets or not self.val_preds:
            return  # Skip if no data
            
        try:
            # Concatenate all predictions and targets
            targets = torch.cat(self.val_targets).numpy()
            preds = torch.cat(self.val_preds).numpy()
            
            # Reset stored predictions and targets for next epoch
            self.val_targets = []
            self.val_preds = []
            
            # Calculate confusion matrix
            cm = confusion_matrix(targets, preds, labels=list(range(self.num_classes)))
            
            # Create and log confusion matrix plot if we have a logger
            if self.logger and hasattr(self.logger, 'experiment'):
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
                ax.set_title(f'Confusion Matrix - Epoch {self.current_epoch}')
                
                # Log to wandb if available
                try:
                    self.logger.experiment.log({
                        "confusion_matrix": wandb.Image(fig),
                        "epoch": self.current_epoch
                    })
                except Exception as e:
                    print(f"Warning: Could not log confusion matrix to logger: {e}")
                
                # Close the figure to free memory
                plt.close(fig)
        except Exception as e:
            print(f"Warning: Error in validation_epoch_end: {e}")

    def configure_optimizers(self):
        # Configure optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Configure learning rate scheduler if specified
        if self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_params.get("T_max", 10),
                eta_min=self.scheduler_params.get("eta_min", 1e-6)
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.scheduler_params.get("mode", "min"),
                factor=self.scheduler_params.get("factor", 0.1),
                patience=self.scheduler_params.get("patience", 5)
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        elif self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_params.get("step_size", 10),
                gamma=self.scheduler_params.get("gamma", 0.1)
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer