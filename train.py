import argparse
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from model.classifier.model import ClassificationModel
from model.classifier.dataset import FundusDataModule
from model.utils.metrics import classification_metrics
import torch.nn as nn
import random
import string

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="efficientnet_b0")
    parser.add_argument("--train_dir", type=str, default="split_dataset/train")
    parser.add_argument("--val_dir", type=str, default="split_dataset/test")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()
      # Set up W&B logger
    unique_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    wandb_logger = WandbLogger(
        project="fundus-classification", 
        save_dir='weight', 
        id=f'{args.run_id}-{unique_id}')

    wandb_logger = WandbLogger(
        project="fundus-classification",
        name=args.model_name,
        log_model="all",
        config=vars(args)
    )


    model = ClassificationModel(
        model_name=args.model_name,
        num_classes=args.num_classes,
        in_channels=3,
        loss_fn=nn.CrossEntropyLoss(),
        metric_fn=classification_metrics,
        lr=args.lr
    )

    datamodule = FundusDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        LearningRateMonitor(logging_interval="epoch")
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        accelerator="auto",
        devices=1,
        strategy="auto"
    )


    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
