import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class ResNetLSTM_MultiModal(nn.Module):
    def __init__(
        self,
        audiomae,
        audio_out_dim = 512,
        backbone_name: str = "resnet18",
        frame_step: int = 2,
        hidden_size: int = 256,
        num_layers: int = 1,
        use_fusion: bool = True,
        use_dropout: bool = False,
        dropout_p: float = 0.3,
        LSTM_bidirectional : bool = True
    ):
        super().__init__()

        # backbones disponíveis: (fn, weights, cnn_out_size)
        backbones = {
            "resnet18": (resnet18, ResNet18_Weights.DEFAULT, 512),
            "resnet34": (resnet34, ResNet34_Weights.DEFAULT, 512),
            "resnet50": (resnet50, ResNet50_Weights.DEFAULT, 2048),
        }

        assert backbone_name in backbones, f"backbone_name deve ser um de {list(backbones.keys())}"

        model_fn, weights, cnn_out_size = backbones[backbone_name]

        backbone = model_fn(weights=weights)

        self.audio_encoder = audiomae.encoder

        for p in self.audio_encoder.parameters():
            p.requires_grad = False
            
        self.audio_encoder.eval()

        self.audio_projection = nn.Linear(768, audio_out_dim) # tentar usar cnn_out_size

        self.frame_step = frame_step # vê um frame a cada frame_step frames

        # arquitetura da ResNet disponível em https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        
        # lista todos os módulos da ResNet e remove o classificador no final, 
        # deixando o AdaptiveAvgPool: (B*T, cnn_out_size, 1, 1)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        # substitui primeiro conv para aceitar grayscale (1 canal), antes era 3 na entrada
        self.cnn[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.use_fusion = use_fusion
        fusion_dim = cnn_out_size + audio_out_dim
        fusion_out_dim = 512

        # BiLSTM processa a sequência de features por frame
        lstm_input_size = fusion_out_dim if use_fusion else fusion_dim
        
        self.lstm = nn.LSTM(
            lstm_input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=LSTM_bidirectional,
            dropout=dropout_p if num_layers > 1 else 0,
        )

        # cabeça de regressão: max pool -> 128 -> 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p) if use_dropout else nn.Identity(),
            nn.Linear(128, 1),
        )

        # projeta o concat de video+audio pra fusion_out_dim antes da LSTM
        if use_fusion:
            self.fusion = nn.Sequential(
                     nn.Linear(fusion_dim, fusion_out_dim),
                     nn.ReLU(),
                     nn.Dropout(dropout_p) if use_dropout else nn.Identity(),
                 )

    # travar sempre o audio_encoder em eval (não vou treinando isso)
    def train(self, mode: bool = True):
        super().train(mode)
        self.audio_encoder.eval() 
        return self

    def forward(self, video, mel):
        # o mel já deve ter sido processado com F.interpolate(mel, size=(128, 1024), mode="bilinear", align_corners=False) para ajustar o shape

        # C vai ser 1 sempre (grayscale)
        # video: (B, T, C, H, W)
        B, T, C, H, W = video.shape

        # subsample temporal
        video = video[:, ::self.frame_step]
        T_sub = video.shape[1]

        # extrai features por frame com a CNN
        video = video.reshape(B*T_sub, C, H, W)
        video = self.cnn(video)           # (B*T_sub, cnn_out_size, 1, 1)
        video = video.view(B, T_sub, -1)  # (B, T_sub, cnn_out_size)

        # audio
        B, T, H_mel, W_mel = mel.shape

        mel = mel.reshape(B * T, 1, H_mel, W_mel)
        mel = F.interpolate(mel, size=(128, 1024), mode='bilinear', align_corners=False)

        mel = mel.transpose(2, 3) # (B * T, 1, 1024, 128)
        
        with torch.no_grad():
            audio_feat = self.audio_encoder.forward_features(mel)

        audio_feat = audio_feat[:, 0] # CLS token

        audio_feat = self.audio_projection(audio_feat) # (B, audio_out_dim)
        audio_feat = audio_feat.view(B, T, -1)

        # repedir o mesmo embedding para todos os frames
        if T == 1:
            audio_feat = audio_feat.expand(-1, T_sub, -1)

        x = torch.cat([video, audio_feat], dim=2) # concatenar informações do video e audio

        if self.use_fusion:
            x = self.fusion(x)

        # processa sequência temporal
        out, _ = self.lstm(x)          # (B, T_sub, hidden_size*2)
        out = out.max(dim=1).values    # (B, hidden_size*2), max pooling sobre T

        return self.head(out)  # (B, 1)
    
'''
O audioMAE pode ser carregado assim:

from transformers import AutoModel # transformers==4.44.0
import einops
import timm
import torchaudio # torchaudio==2.5.1

model_ae = AutoModel.from_pretrained("hance-ai/audiomae", trust_remote_code=True).to(device)
'''