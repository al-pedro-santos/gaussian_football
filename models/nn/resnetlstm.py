import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetLSTM(nn.Module):
    def __init__(
        self,
        frame_step=5,
        hidden_size=256,
        num_layers=2,
        use_dropout=False,
        dropout_p=0.3,
    ):
        super().__init__()

        self.frame_step = frame_step

        # extrator de features por frame
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # remove a camada de classificação final
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # (B*T, 512, 1, 1)

        # primeiro conv para 1 canal (grayscale)
        self.cnn[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        cnn_out_size = 512

        self.lstm = nn.LSTM(
            cnn_out_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_p if num_layers > 1 else 0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p) if use_dropout else nn.Identity(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # subsample de frames
        x = x[:, :: self.frame_step, :, :, :]
        T_sub = x.shape[1]

        # CNN em cada frame
        x = x.reshape(B * T_sub, C, H, W)
        x = self.cnn(x)  # (B*T_sub, 512, 1, 1)
        x = x.view(B, T_sub, -1)  # (B, T_sub, 512)

        # sequência temporal
        out, (h_n, _) = self.lstm(x)  # h_n: (num_layers*2, B, hidden)
        # pega os últimos hidden states das duas direções
        h_fwd = h_n[-2]  # (B, hidden)
        h_bwd = h_n[-1]  # (B, hidden)
        out = torch.cat([h_fwd, h_bwd], dim=1)  # (B, hidden*2)

        return self.head(out)  # (B, 1)
