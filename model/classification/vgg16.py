import torch
import torch.nn as nn
from torchvision.models import vgg16

__all__ = [ "VGG16" ]

class VGG16(nn.Module):
    """VGG16 pretrained on imagent (Baseline model)"""
    def __init__(self, n_classes):
        super().__init__()
        backbone = vgg16(pretrained=True)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)


if __name__ == "__main__":
    from torchsummary import summary

    model = VGG16(n_classes=50)
    summary(model, (3, 32, 32), device="cpu")
    print(model)
