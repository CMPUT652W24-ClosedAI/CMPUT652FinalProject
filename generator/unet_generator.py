import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.downsample = torch.nn.MaxPool2d(2)

        self.block_1 = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding="same"),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding="same"),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding="same"),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding="same"),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding="same"),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        self.upsample_1 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.up_block_1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding="same"),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.upsample_2 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.up_block_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding="same"),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        self.output_layer = nn.Conv2d(64, 6, 1, padding="same")

    def forward(self, x):
        h1 = self.block_1(x)
        d1 = self.downsample(h1)
        h2 = self.block_2(d1)
        d2 = self.downsample(h2)
        h3 = self.block_3(d2)

        u1 = self.upsample_1(h3)
        c1 = torch.cat((h2, u1), dim=-3)
        h4 = self.up_block_1(c1)
        u2 = self.upsample_2(h4)
        c2 = torch.cat((h1, u2), dim=-3)
        h5 = self.up_block_2(c2)
        output = self.output_layer(h5)
        return output
