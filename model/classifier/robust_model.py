import lightning.pytorch as pl
import torch
import torch.nn as nn
import timm
import torchmetrics
import numpy as np


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
        
        # Set loss function
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        
        # Custom metric function (optional)
        self.metric_fn = metric_fn
        
        # Training hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}
        
        # Initialize metrics - only essential ones
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        # For optional confusion matrix
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)
    
    def _process_batch(self, batch):
        """Safely process batch data regardless of format"""
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                return batch[0], batch[1]
            elif len(batch) > 2:
                # If more than 2 elements, take first two
                print(f"Warning: Batch has {len(batch)} elements, using first two")
                return batch[0], batch[1]
            else:
                raise ValueError(f"Batch has only {len(batch)} elements, expected at least 2")
        elif isinstance(batch, dict):
            # Handle dict format
            if 'image' in batch and 'label' in batch:
                return batch['image'], batch['label']
            elif 'images' in batch and 'labels' in batch:
                return batch['images'], batch['labels']
            else:
                raise ValueError(f"Dict batch missing expected keys: {batch.keys()}")
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

    def training_step(self, batch, batch_idx):
        try:
            x, y = self._process_batch(batch)
            logits = self(x)
            loss = self.loss_fn(logits, y)
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            acc = self.train_acc(preds, y)
            
            # Log only essential metrics
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

            return loss
        except Exception as e:
            print(f"Error in training_step: {e}")
            print(f"Batch type: {type(batch)}")
            if isinstance(batch, (list, tuple)):
                print(f"Batch length: {len(batch)}")
            raise

    def validation_step(self, batch, batch_idx):
        try:
            x, y = self._process_batch(batch)
            logits = self(x)
            loss = self.loss_fn(logits, y)
            
            # Calculate predictions
            preds = torch.argmax(logits, dim=1)
            
            # Update accuracy
            acc = self.val_acc(preds, y)
            
            # Log only essential metrics
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

            return {"val_loss": loss}
        except Exception as e:
            print(f"Error in validation_step: {e}")
            print(f"Batch type: {type(batch)}")
            if isinstance(batch, (list, tuple)):
                print(f"Batch length: {len(batch)}")
            raise

    def configure_optimizers(self):
        # Configure optimizer
        if hasattr(self.hparams, 'optimizer'):
            opt_name = self.hparams.optimizer
        else:
            opt_name = 'adamw'
            
        if opt_name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif opt_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif opt_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
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
                patience=self.scheduler_params.get("patience", 5),
                min_lr=1e-6
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
            
    def on_validation_epoch_end(self):
        """Empty by default - no confusion matrix to save storage"""
        pass