import torch
import torch.nn as nn
from torchvision.models import resnet50


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

        res50 = resnet50(pretrained=True)
        self.res = nn.Sequential(*list(res50.children())[:-2])

        self.deconv1 = nn.ConvTranspose2d(2048, 1024, 2, 2, 0)
        self.norm1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        self.norm2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 2, 2, 0)
        self.norm3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        self.norm4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.norm5 = nn.BatchNorm2d(64)
        self.conv = nn.Conv2d(64, 1, 1, 1, 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.cat(
            (torch.cat((x[:, 0], x[:, 1], x[:, 2]), dim=3),
             torch.cat((x[:, 3], x[:, 4], x[:, 5]), dim=3)), dim=2)

        x = self.res(x)
        x = self.relu(self.norm1(self.deconv1(x)))
        x = self.relu(self.norm2(self.deconv2(x)))
        x = self.relu(self.norm3(self.deconv3(x)))
        x = self.relu(self.norm4(self.deconv4(x)))
        x = self.relu(self.norm5(self.deconv5(x)))
        x = self.conv(x)
        return x
