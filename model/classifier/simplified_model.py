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
                 lr=1e-3,
                 weight_decay=1e-4,
                 scheduler_type=None,
                 scheduler_params=None,
                 simplified_logging=True):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn'])
        
        # Initialize model from timm
        self.model = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=num_classes, 
            in_chans=in_channels
        )
        
        # Set loss function or default to CrossEntropyLoss
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        
        # Training hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}
        
        # Simplified logging flag
        self.simplified_logging = simplified_logging
        
        # Initialize metrics for tracking
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        # For confusion matrix (only used for final evaluation)
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
        
        # Log only the essential metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Calculate predictions
        preds = torch.argmax(logits, dim=1)
        
        # Store targets and predictions for final confusion matrix
        # Only do this in the last epoch to save memory
        if self.current_epoch == self.trainer.max_epochs - 1:
            self.val_targets.append(y.cpu())
            self.val_preds.append(preds.cpu())
        
        # Update metrics
        acc = self.val_acc(preds, y)
        
        # Log only the essential metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return {"val_loss": loss, "val_preds": preds, "val_targets": y}

    def on_validation_epoch_end(self):
        """Calculate and log confusion matrix only at the end of training"""
        # Skip confusion matrix logging during training if using simplified logging
        if self.simplified_logging and self.current_epoch < self.trainer.max_epochs - 1:
            self.val_targets = []
            self.val_preds = []
            return
            
        # Process the stored predictions only if we have any
        if not self.val_targets or not self.val_preds:
            return
            
        try:
            # Concatenate all predictions and targets
            targets = torch.cat(self.val_targets).numpy()
            preds = torch.cat(self.val_preds).numpy()
            
            # Reset stored predictions and targets for next epoch
            self.val_targets = []
            self.val_preds = []
            
            # Only log confusion matrix in the final epoch if using simplified logging
            if self.simplified_logging and self.current_epoch < self.trainer.max_epochs - 1:
                return
                
            # Calculate confusion matrix
            cm = confusion_matrix(targets, preds, labels=list(range(self.num_classes)))
            
            # Create and log confusion matrix plot if we have a logger
            if self.logger and hasattr(self.logger, 'experiment'):
                # Use smaller figure size and lower DPI to reduce storage usage
                fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
                ax.set_title(f'Confusion Matrix - Final Epoch')
                
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