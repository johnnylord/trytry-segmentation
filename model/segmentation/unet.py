import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import vgg16


def double_conv(in_channels, out_channels):
    layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
    return layer


class UNet(nn.Module):
    """UNet trained from scratch without pretrained on imagenet

    This one will fail to pass the baseline. Pretrained weights from imagenet
    seems to be a very IMPORTANT factor to build a good model.

    See `UNetVGG16` in below to have a segmentation model with unet architecture
    where the encoder part is a vgg16 feature layer pretrained on imagenet.
    """
    def __init__(self, n_classes):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        # ===================================================
        # (N, 3, 512, 512)
        self.conv1 = double_conv(3, 64)
        self.down_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        # (N, 64, 256, 256)
        self.conv2 = double_conv(64, 128)
        self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        # (N, 128, 128, 128)
        self.conv3 = double_conv(128, 256)
        self.down_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        # (N, 256, 64, 64)
        self.conv4 = double_conv(256, 512)
        self.down_conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        # (N, 512, 32, 32)
        self.conv5 = double_conv(512, 1024)

        # Decoder
        # ====================================================
        # (N, 1024, 32, 32)
        self.up_conv6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = double_conv(1024, 512)
        # (N, 512, 64, 64)
        self.up_conv7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = double_conv(512, 256)
        # (N, 256, 128, 128)
        self.up_conv8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = double_conv(256, 128)
        # (N, 128, 256, 256)
        self.up_conv9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = double_conv(128, 64)

        # Classifier
        # ===================================================
        # (N, 64, 512, 512)
        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

        self.apply(self._init_weight)

    def forward(self, x):
        # Encoder
        # =======================
        x1 = self.conv1(x)
        x = self.down_conv1(x1)
        x2 = self.conv2(x)
        x = self.down_conv2(x2)
        x3 = self.conv3(x)
        x = self.down_conv3(x3)
        x4 = self.conv4(x)
        x = self.down_conv4(x4)
        x5 = self.conv5(x)

        # Decoder
        # =======================
        x = self.up_conv6(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv9(x)

        x = self.classifier(x)
        return x

    def _init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            init.normal_(m.weight.data)
            init.zeros_(m.bias.data)


class UNetVGG16(nn.Module):
    """UNet with pretrained vgg16 as encoder part"""
    def __init__(self, n_classes):
        super().__init__()
        backbone = vgg16(pretrained=True)
        features = backbone.features
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        # =============================
        # (3, 512, 512)
        self.conv1 = features[0:4]
        # (64, 256, 256)
        self.conv2 = features[5:9]
        # (128, 128, 128)
        self.conv3 = features[10:16]
        # (256, 64, 64)
        self.conv4 = features[17:23]
        # (512, 32, 32)
        self.conv5 = features[24:30]

        # Decoder
        # =============================
        # (512, 32, 32)
        self.up_conv6 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv6 = double_conv(1024, 512)
        # (512, 64, 64)
        self.up_conv7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = double_conv(512, 256)
        # (256, 128, 128)
        self.up_conv8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = double_conv(256, 128)
        # (128, 256, 256)
        self.up_conv9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = double_conv(128, 64)

        # Classifier
        # ===============================
        # (64, 512, 512)
        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x = self.max_pool(x1)
        x2 = self.conv2(x)
        x = self.max_pool(x2)
        x3 = self.conv3(x)
        x = self.max_pool(x3)
        x4 = self.conv4(x)
        x = self.max_pool(x4)
        x = self.conv5(x)

        # Decoder
        x = torch.cat([x4, self.up_conv6(x)], dim=1)
        x = self.conv6(x)
        x = torch.cat([x3, self.up_conv7(x)], dim=1)
        x = self.conv7(x)
        x = torch.cat([x2, self.up_conv8(x)], dim=1)
        x = self.conv8(x)
        x = torch.cat([x1, self.up_conv9(x)], dim=1)
        x = self.conv9(x)

        # Classifier
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    from torchsummary import summary

    model = UNetVGG16(n_classes=7).eval()
    summary(model, (3, 512, 512), device="cpu")
