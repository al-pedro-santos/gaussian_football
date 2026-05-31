import torch
import torch.nn as nn
from torchvision import models

class VGG16_LSTM(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        # Carrega VGG16 pré-treinado sem a cabeça
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.cnn_base = vgg.features  # output: (B*T, 512, 7, 7) para input 224x224

        # Congela tudo até block5_conv1
        # No PyTorch, block5_conv1 equivale ao índice 24 nas features do VGG16
        for i, layer in enumerate(self.cnn_base.children()):
            if i < 24:
                for param in layer.parameters():
                    param.requires_grad = False

        self.gap = nn.AdaptiveAvgPool2d(1)  # (B*T, 512, 1, 1)

        cnn_out_size = 512

        self.lstm1 = nn.LSTM(cnn_out_size, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 100, batch_first=True)
        self.num_frames = num_frames

    def forward(self, x):
        # x: (B, T, C, H, W) — T=num_frames, C=3, H=W=224
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        x = self.cnn_base(x)          # (B*T, 512, 7, 7)
        x = self.gap(x)               # (B*T, 512, 1, 1)
        x = x.view(B, T, -1)          # (B, T, 512)

        out, _ = self.lstm1(x)        # (B, T, 256)
        out, (h_n, _) = self.lstm2(out)
        return h_n[-1]                # (B, 100)