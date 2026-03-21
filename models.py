# models.py
import torch
import torch.nn as nn
import torchvision.models as models

class SmallConvAE(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, in_ch, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))

def _adapt_first_conv(conv: nn.Conv2d, in_ch: int) -> nn.Conv2d:

    assert conv.in_channels == 3
    new_conv = nn.Conv2d(in_ch, conv.out_channels, kernel_size=conv.kernel_size,
                         stride=conv.stride, padding=conv.padding, bias=False)
    with torch.no_grad():
        w = conv.weight  
        new_conv.weight[:, :3] = w
        if in_ch > 3:
            mean_w = w.mean(dim=1, keepdim=True)
            for c in range(3, in_ch):
                new_conv.weight[:, c:c+1] = mean_w
    return new_conv

class ResNet50_6ch(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.conv1 = _adapt_first_conv(m.conv1, in_ch=in_ch)
        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat).squeeze(1)
