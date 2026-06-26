import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, num_frames, num_input_tokens=25088):
        super().__init__()
        # Bidirectional LSTM(200) -> output is 400 (200 * 2 directions)
        self.bilstm = nn.LSTM(num_input_tokens, 200, batch_first=True, bidirectional=True)
        self.lstm2   = nn.LSTM(400, 100, batch_first=True)
        self.drop2   = nn.Dropout(0.3)
        self.lstm3   = nn.LSTM(100, 100, batch_first=True)
        self.drop3   = nn.Dropout(0.3)

    def forward(self, x):
        out, _ = self.bilstm(x)          # (B, T, 400)
        out = self.drop2(out)            # dropout do lstm2
        out, _ = self.lstm2(out)         # (B, T, 100)
        out = self.drop3(out)            # dropout do lstm3
        out, (h_n, _) = self.lstm3(out)  # (B, T, 100)
        return h_n[-1]