import torch
import torch.nn as nn
import torch.nn.functional as F

# The codes are borrowed and modified from :
# https://github.com/milesial/Pytorch-UNet/tree/master/unet

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bias=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv_blocks = nn.ModuleList()
        self.relu = nn.ReLU()
        in_temp = in_channels
        out_temp = out_channels // 2  # //4
        self.drop = nn.Dropout(0.5)
        for i in range(4):
            self.conv_blocks.append(nn.Conv2d(in_temp, out_temp, kernel_size=3, padding=1, bias=bias))
            in_temp = in_temp + out_temp

        self.out_conv = nn.Conv2d(in_temp, out_channels, kernel_size=1, padding=0, bias=bias)
        # self.bn = nn.BatchNorm2d(out_channels)  # Batch Normalization

    def forward(self, x):
        for block in self.conv_blocks:
            x = torch.cat([x, self.relu(block(x))], 1)
        return self.relu(self.out_conv(x))

class DoubleConv_dropout(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bias=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv_blocks = nn.ModuleList()
        self.relu = nn.ReLU()
        in_temp = in_channels
        out_temp = out_channels // 2  # //4
        self.drop = nn.Dropout(0.5)
        for i in range(4):
            self.conv_blocks.append(nn.Conv2d(in_temp, out_temp, kernel_size=3, padding=1, bias=bias))
            in_temp = in_temp + out_temp

        self.out_conv = nn.Conv2d(in_temp, out_channels, kernel_size=1, padding=0, bias=bias)
        # self.bn = nn.BatchNorm2d(out_channels)  # Batch Normalization

    def forward(self, x):
        for block in self.conv_blocks:
            x = torch.cat([x, self.relu(block(x))], 1)
        return self.drop(self.relu(self.out_conv(x)))



class Down(nn.Module):
    """Down-scaling with max pooling then DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Up-scaling then DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):  # x2: copied feature maps
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up_dropout(nn.Module):
    """Up-scaling then DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_dropout(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_dropout(in_channels, out_channels)

    def forward(self, x1, x2):  # x2: copied feature maps
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class OutConv(nn.Module):
    """Map to the desired number of channels"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1 convolution

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(UNet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.inc = DoubleConv(self.in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.out_ch)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        a = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        # x = x
        # x = self.relu(x)  ## dongkeun

        return x

from model.transformer import TransformerBlock

class UNet_transformer(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(UNet_transformer, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.transformer1 = TransformerBlock(channels=64, num_heads=1, expansion_factor=0.5)
        self.transformer2 = TransformerBlock(channels=64*2, num_heads=1, expansion_factor=0.5)
        self.transformer3 = TransformerBlock(channels=64*4, num_heads=1, expansion_factor=0.5)
        self.transformer4 = TransformerBlock(channels=64*8, num_heads=1, expansion_factor=0.5)
        self.transformer5 = TransformerBlock(channels=64*16, num_heads=1, expansion_factor=0.5)

        self.inc = DoubleConv(self.in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.out_ch)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        a = x
        x1 = self.transformer1(self.inc(x))
        x2 = self.transformer2(self.down1(x1))
        x3 = self.transformer3(self.down2(x2))
        x4 = self.transformer4(self.down3(x3))
        x5 = self.transformer5(self.down4(x4))
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x = self.outc(x9)

        # x = x
        # x = self.relu(x)  ## dongkeun

        return x,x1,x2,x3,x4,x5,x6,x7,x8,x9

class UNet_encoder(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(UNet_encoder, self).__init__()
        self.in_ch = in_ch

        self.inc = DoubleConv(self.in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5

class UNet_for_CC_decoder_dropout(nn.Module):
    def __init__(self, out_ch, bilinear=False):
        super(UNet_for_CC_decoder_dropout, self).__init__()

        self.out_ch = out_ch

        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.dropout = nn.Dropout(p=0.5)
        self.outc = OutConv(256, self.out_ch)
        # self.conv_1x1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1,stride=1, padding=0)
        # self.conv_1x1_2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, rgb_5, rgb_4, rgb_3):
        x = self.up1(rgb_5, rgb_4)
        x = self.up2(x, rgb_3)
        x = self.dropout(x)
        out = self.relu(self.outc(x))

        return out
