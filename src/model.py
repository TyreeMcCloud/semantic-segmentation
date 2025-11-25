import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.uconv4 = DoubleConv(1024, 512)

        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.uconv3 = DoubleConv(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.uconv2 = DoubleConv(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.uconv1 = DoubleConv(128, 64)

        self.output = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))

        bn = self.bottleneck(self.pool(c4))

        u4 = self.u4(bn)
        u4 = self.uconv4(torch.cat([u4, c4], dim=1))

        u3 = self.u3(u4)
        u3 = self.uconv3(torch.cat([u3, c3], dim=1))

        u2 = self.u2(u3)
        u2 = self.uconv2(torch.cat([u2, c2], dim=1))

        u1 = self.u1(u2)
        u1 = self.uconv1(torch.cat([u1, c1], dim=1))

        return self.output(u1)
