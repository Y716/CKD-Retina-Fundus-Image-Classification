import lightning.pytorch as pl
import torch
import timm

class ClassificationModel(pl.LightningModule):
    def __init__(self, 
                 model_name, 
                 num_classes,
                 in_channels=3,
                 loss_fn=None,
                 metric_fn=None,
                 lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, in_chans=in_channels)
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if self.metric_fn:
            metrics = self.metric_fn(logits, y)
            metrics = {f"train_{k}": v for k, v in metrics.items()}  # Prefix with "train_"
            self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if self.metric_fn:
            metrics = self.metric_fn(logits, y)
            metrics = {f"val_{k}": v for k, v in metrics.items()}  # Prefix with "val_"
            self.log_dict(metrics, prog_bar=True,  on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
