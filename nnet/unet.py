from collections import OrderedDict

import torch
import torch.nn as nn

from .misc import summary


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, features: int, name: str):
        super().__init__()
        self.in_channels = in_channels
        self.features = features
        self.name = name
        self.conv1 = (
            f"{name}_conv1",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )
        self.conv2 = (
            f"{name}_conv2",
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )
        self.norm1 = (f"{name}_norm1", nn.BatchNorm2d(num_features=features))
        self.norm2 = (f"{name}_norm2", nn.BatchNorm2d(num_features=features))
        self.relu1 = (f"{name}_relu1", nn.ReLU())
        self.relu2 = (f"{name}_relu2", nn.ReLU())

        self.enc = nn.Sequential(
            OrderedDict(
                [
                    self.conv1,
                    self.norm1,
                    self.relu1,
                    self.conv2,
                    self.norm2,
                    self.relu2,
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        return self.enc(x)


class Small_UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()

        self.feats = init_features
        # NOTE: Encoder module
        self.encoder1 = DoubleConv(in_channels, self.feats, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(self.feats, self.feats * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv(self.feats * 2, self.feats * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.encoder4 = DoubleConv(self.feats * 4, self.feats * 8, name="enc4")
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Middle bottle-neck
        self.bottleneck = DoubleConv(
            self.feats * 4, self.feats * 8, name="bottleneck"
        )

        # self.upconv4 = nn.ConvTranspose2d(
        #     self.feats * 16, self.feats * 8, kernel_size=2, stride=2
        # )
        # self.decoder4 = DoubleConv(
        #     (self.feats * 8) * 2, self.feats * 8, name="dec4"
        # )
        self.upconv3 = nn.ConvTranspose2d(
            self.feats * 8, self.feats * 4, kernel_size=2, stride=2
        )
        self.decoder3 = DoubleConv(
            (self.feats * 4) * 2, self.feats * 4, name="dec3"
        )
        self.upconv2 = nn.ConvTranspose2d(
            self.feats * 4, self.feats * 2, kernel_size=2, stride=2
        )
        self.decoder2 = DoubleConv(
            (self.feats * 2) * 2, self.feats * 2, name="dec2"
        )
        self.upconv1 = nn.ConvTranspose2d(
            self.feats * 2, self.feats, kernel_size=2, stride=2
        )
        self.decoder1 = DoubleConv(self.feats * 2, self.feats, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=self.feats, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        # enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc3))

        # dec4 = self.upconv4(bottleneck)
        # dec4 = torch.cat((dec4, enc4), dim=1)
        # dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        self.feats = init_features
        # NOTE: Encoder module
        self.encoder1 = DoubleConv(in_channels, self.feats, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(self.feats, self.feats * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv(self.feats * 2, self.feats * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = DoubleConv(self.feats * 4, self.feats * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Middle bottle-neck
        self.bottleneck = DoubleConv(
            self.feats * 8, self.feats * 16, name="bottleneck"
        )

        self.upconv4 = nn.ConvTranspose2d(
            self.feats * 16, self.feats * 8, kernel_size=2, stride=2
        )
        self.decoder4 = DoubleConv(
            (self.feats * 8) * 2, self.feats * 8, name="dec4"
        )
        self.upconv3 = nn.ConvTranspose2d(
            self.feats * 8, self.feats * 4, kernel_size=2, stride=2
        )
        self.decoder3 = DoubleConv(
            (self.feats * 4) * 2, self.feats * 4, name="dec3"
        )
        self.upconv2 = nn.ConvTranspose2d(
            self.feats * 4, self.feats * 2, kernel_size=2, stride=2
        )
        self.decoder2 = DoubleConv(
            (self.feats * 2) * 2, self.feats * 2, name="dec2"
        )
        self.upconv1 = nn.ConvTranspose2d(
            self.feats * 2, self.feats, kernel_size=2, stride=2
        )
        self.decoder1 = DoubleConv(self.feats * 2, self.feats, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=self.feats, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))


if __name__ == "__main__":
    model = Small_UNet(init_features=2)
    summary(model)
    pass
