from model import Net
import os
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy


class LitResnet(LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.model = Net()
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.59E-02, epochs=self.trainer.max_epochs, total_steps=self.trainer.estimated_stepping_batches, anneal_strategy='cos',\
                                                div_factor=10, pct_start=(5/self.trainer.max_epochs), three_phase=False)

        return ([optimizer], [scheduler])