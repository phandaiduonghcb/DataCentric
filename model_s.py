import pytorch_lightning as pl
# import timm
import torch
import torch.nn as nn
import torchmetrics
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
import wandb
import logging

from networks import Resnet
log = logging.getLogger(__name__)
class TIMMModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.criterion = instantiate(self.config.loss)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10, top_k=1)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10, top_k=1)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10, top_k=1)
        self.model = instantiate(self.config.arch)
        self.test_table = []
        self.predicted_targets = []

    def configure_optimizers(self):
        optimizer = instantiate(self.config.optimizer, params=self.model.parameters())
        scheduler = instantiate(self.config.lr_scheduler, optimizer=optimizer)
        if "ReduceLROnPlateau" in self.config.lr_scheduler._target_:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.config.monitor,
            }
        return [optimizer], [scheduler]

    def forward(self,x):
        x = self.model(x)
        return x
    
    def step(self, batch):
        x,y = batch[:2]
        logits = self.forward(x)
        loss = self.criterion(logits,y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        acc = self.train_acc(preds,targets)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.config.sync_dist,
        )
        self.log(
            "train/acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.config.sync_dist,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> None:
        val_loss, val_preds, val_targets = self.step(batch)
        val_acc = self.val_acc(val_preds, val_targets)
        self.log(
            "val/loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.config.sync_dist,
        )
        self.log(
            "val/acc",
            val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.config.sync_dist,
        )
        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx) -> None:
        test_loss, test_preds, test_targets = self.step(batch)
        test_acc = self.test_acc(test_preds, test_targets)

        mask = (test_preds != test_targets)
        idxs = torch.argwhere(mask)

        for i in idxs:
            self.test_table.append([str(test_targets[i].item()+1), str(test_preds[i].item()+1), wandb.Image(batch[0][i]), batch[2][i]])
            
        self.log(
            "test/loss",
            test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.config.sync_dist,
        )
        self.log(
            "test/acc",
            test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.config.sync_dist,
        )
        return {"test_loss": test_loss}
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, targets = self.step(batch)
        self.predicted_targets.append(targets)
        return preds
    
    def on_epoch_end(self):
        self.train_acc.reset()
        self.val_acc.reset()

    def on_test_end(self):
        columns = ["true","pred", "image","path"]
        # [build up your predictions data as above]
        self.logger.log_table(key='wrong_pred_images', columns=columns, data= self.test_table)
    
    def on_predict_end(self) -> None:
        self.predicted_targets = torch.cat(self.predicted_targets)