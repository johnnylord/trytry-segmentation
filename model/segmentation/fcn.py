import torch
import torch.nn as nn
from torchvision.models import vgg16


__all__ = [ "FCN32" ]

class FCN32(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        backbone = vgg16(pretrained=True)

        # Pretrained feature extractor of vgg16 on imagenet
        self.features = backbone.features

        self.classifier = nn.Sequential(
                            # (512, 16, 16)
                            nn.Conv2d(512, 4096, kernel_size=1),
                            nn.ReLU(inplace=True),
                            nn.Dropout2d(),
                            # (4096, 16, 16)
                            nn.Conv2d(4096, 4096, kernel_size=1),
                            nn.ReLU(inplace=True),
                            nn.Dropout2d(),
                            # (4096, 16, 16)
                            nn.Conv2d(4096, n_classes, kernel_size=1),
                            )

        # (n_classes, 16, 16)
        self.upsample = nn.ConvTranspose2d(n_classes, n_classes, 32, 32)

    def forward(self, x):
        # x shape: (512, 512)
        x = self.features(x)
        x = self.classifier(x)
        mask = self.upsample(x)
        return mask


if __name__ == "__main__":
    from torchsummary import summary

    model = FCN32(n_classes=7).eval()
    summary(model, (3, 512, 512), device="cpu")
