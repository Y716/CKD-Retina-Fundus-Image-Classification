import lightning.pytorch as pl
import torch
from torchvision.utils import make_grid

class SegmentationModel(pl.LightningModule):
    def __init__(self, unet, loss, metrics_fn=None, optimizer_config=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = unet
        self.loss_fn = loss
        self.get_metrics = metrics_fn
        self.optimizer_config = optimizer_config
        
        self.validation_step_outputs = {'image': [], 'mask': [], 'pred': []}
        
    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, idx):
        _, x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log('train_loss', loss.item(), prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, idx):
        r, x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log('val_loss', loss.item(), prog_bar=True, logger=True)
        if self.get_metrics:
          metrics = self.get_metrics(pred, y)
          self.log_dict(metrics, logger=True)
        
        self.validation_step_outputs['image'].extend(r)
        self.validation_step_outputs['mask'].extend(y)
        self.validation_step_outputs['pred'].extend(pred)
        return loss
    
    def on_validation_epoch_end(self):
        results = zip(self.validation_step_outputs['image'], self.validation_step_outputs['mask'], self.validation_step_outputs['pred'])
        image_results = []
        for x, y, pred in results:
            mask = torch.cat([y, y, y], dim=0)
            pred = torch.sigmoid(pred)
            pred = torch.where(pred >= .5, 1, 0)
            pred = torch.cat([pred, pred, pred], dim=0)
            image_results.append(make_grid([x, mask, pred], 3))
        
        self.logger.log_image(key='segmentation_result', images=image_results, caption=[str(i) for i in range(len(image_results))])
        
        self.validation_step_outputs['image'].clear()
        self.validation_step_outputs['mask'].clear()
        self.validation_step_outputs['pred'].clear()
    
    def configure_optimizers(self):
        return self.optimizer_config and self.optimizer_config(self.parameters())
      
    @staticmethod
    def from_pretrained(chekpoint):
        SegmentationModel.load_from_checkpoint(checkpoint_path=chekpoint)