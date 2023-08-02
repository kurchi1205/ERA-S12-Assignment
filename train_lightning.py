from model import Net
import os
import torch

class LitResnet(LightningModule):
    def __init__(self, ):
        self.model = Net()

    def forward(self, x):
        return self.model(x)
    


