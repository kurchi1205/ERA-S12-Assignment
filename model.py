import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_channels = 10
        self.prepblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.resblock1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1)
        )
        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.1),)
        self.pool2 = nn.MaxPool2d(2, stride=2)


        self.convblock3 = nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1, bias=False),
                    nn.ReLU(),
                    nn.BatchNorm2d(512),
                    nn.Dropout(0.1),)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.resblock2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(0.1)
        )

        self.pool4 = nn.MaxPool2d(4, stride=2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.prepblock1(x)
        x = self.pool1(self.convblock1(x))
        r1 = self.resblock1(x)
        x = x + r1
        x = self.pool2(self.convblock2(x))
        x = self.pool3(self.convblock3(x))
        r2 = self.resblock2(x)
        x = x+r2
        x = self.pool4(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)