import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(
        self,
        num_frames,
        in_channels=3,
        use_dropout=False,
        use_batch_norm=False,
        dropout_p=0.3,
    ):
        """
        Args:
            num_frames:      número de frames por clipe (T)
            in_channels:     canais da imagem (3 = RGB)
            use_dropout:     ativa dropout nas LSTMs e no head
            use_batch_norm:  ativa batch norm na CNN e no head
            dropout_p:       probabilidade base do dropout
        """
        super().__init__()

        # ------------------------------------------------------------------ #
        # CNN aplicada em cada frame individualmente
        # independente do image_size recebido
        # ------------------------------------------------------------------ #
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

        self.cnn = nn.Sequential(
            conv_block(in_channels, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 512),
            nn.AdaptiveAvgPool2d((1, 1)),   # (B*T, 512, 1, 1)
            nn.Flatten(),                   # (B*T, 512)
        )
        cnn_out_size = 512

        # ------------------------------------------------------------------ #
        # LSTMs — entende a sequência temporal entre frames
        # BiLSTM lê o clipe nos dois sentidos → saída dobra (512 * 2 = 1024)
        # ------------------------------------------------------------------ #
        self.bilstm = nn.LSTM(
            cnn_out_size, 256,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(512, 128, batch_first=True)
        self.lstm3 = nn.LSTM(128, 128, batch_first=True)

        # ------------------------------------------------------------------ #
        # Head — converte a representação final em score por clipe
        # ------------------------------------------------------------------ #
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_p) if use_dropout else nn.Identity(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.5) if use_dropout else nn.Identity(),
            nn.Linear(64, 1),   # score final do clipe
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # aplica CNN em cada frame
        x = x.view(B * T, C, H, W)     # (B*T, C, H, W)
        x = self.cnn(x)                 # (B*T, 512)
        x = x.view(B, T, -1)            # (B, T, 512)

        # sequência temporal
        out, _        = self.bilstm(x)          # (B, T, 512)
        out, _        = self.lstm2(out)          # (B, T, 128)
        _, (h_n, _)   = self.lstm3(out)          # h_n: (1, B, 128)
        out           = h_n[-1]                  # (B, 128)

        # score final
        return self.head(out)                    # (B, 1)