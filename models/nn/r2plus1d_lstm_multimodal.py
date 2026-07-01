import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class R2Plus1DLSTM_MultiModal(nn.Module):
    """
    Versão multimodal (2+1)D CNN (R2Plus1D-18, pré-treinada na Kinetics-400).

    Pipeline:
        vídeo -> R(2+1)D-18 (sem avgpool/fc) -> pooling espacial -> (B, T', 512)
        áudio -> AudioMAE (congelado) -> CLS token -> projeção (Linear, com
                 dropout opcional) -> (B, T_audio, audio_out_dim)
        concat(vídeo, áudio) -> projeção de fusão (Linear + ReLU + Dropout)
        -> BiLSTM -> max pooling temporal -> cabeça de regressão -> (B, 1)
    """

    def __init__(
        self,
        audiomae,
        audio_out_dim: int = 512,
        hidden_size: int = 256,
        num_layers: int = 1,
        use_fusion: bool = True,
        use_dropout: bool = True,
        dropout_p: float = 0.3,
        LSTM_bidirectional: bool = True,
        frame_step: int = 1,
        in_channels: int = 1, # 1 = grayscale
        pretrained: bool = True,
        use_grad_checkpoint: bool = False,
    ):
        super().__init__()

        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        backbone = r2plus1d_18(weights=weights)

        self.cnn = nn.Sequential(
            backbone.stem,    # índice 0
            backbone.layer1,  # índice 1
            backbone.layer2,  # índice 2
            backbone.layer3,  # índice 3
            backbone.layer4,  # índice 4
        )
        cnn_out_size = 512  # canais de saída do layer4 da r2plus1d_18

        # adapta o primeiro Conv3d do stem para aceitar in_channels (grayscale
        # por padrão). O stem do R(2+1)D é: Conv3d(3,45,(1,7,7)) -> BN -> ReLU
        # -> Conv3d(45,64,(3,1,1)) -> BN -> ReLU. Só o primeiro conv recebe os
        # canais de entrada do vídeo, então só ele precisa ser trocado.
        if in_channels != 3:
            old_conv = self.cnn[0][0]
            new_conv = nn.Conv3d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            if pretrained:
                with torch.no_grad():
                    if in_channels == 1:
                        # soma os pesos dos 3 canais RGB -> 1 canal grayscale
                        new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
                    else:
                        c = min(in_channels, 3)
                        new_conv.weight[:, :c].copy_(old_conv.weight[:, :c])
            self.cnn[0][0] = new_conv

        self.frame_step = frame_step
        self.use_grad_checkpoint = use_grad_checkpoint

        # Áudio (AudioMAE congelado)
        self.audio_encoder = audiomae.encoder
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        self.audio_encoder.eval()

        self.audio_projection = nn.Sequential(
            nn.Linear(768, audio_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p) if use_dropout else nn.Identity(),
        )

        self.use_fusion = use_fusion
        fusion_dim = cnn_out_size + audio_out_dim
        fusion_out_dim = 512

        lstm_input_size = fusion_out_dim if use_fusion else fusion_dim

        self.lstm = nn.LSTM(
            lstm_input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=LSTM_bidirectional,
            dropout=dropout_p if num_layers > 1 else 0,
        )

        lstm_out_dim = hidden_size * (2 if LSTM_bidirectional else 1)

        # cabeça de regressão: max pool -> 128 -> 1
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
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

    def _forward_cnn(self, x):
        """Roda self.cnn, opcionalmente com gradient checkpointing por bloco
        (stem + layer1..4) para reduzir VRAM — útil pois convoluções 3D
        consomem bem mais memória que as 2D da ResNet."""
        if self.use_grad_checkpoint and self.training:
            for block in self.cnn:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            x = self.cnn(x)
        return x

    def forward(self, video, mel):
        # video: (B, T, C, H, W), C deve ser igual a in_channels (1 = grayscale)
        if self.frame_step > 1:
            video = video[:, ::self.frame_step]

        # Conv3d espera (B, C, T, H, W)
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        video = self._forward_cnn(video) # (B, 512, T', H', W')
        video = video.mean(dim=[3, 4]) # pooling espacial -> (B, 512, T')
        video = video.transpose(1, 2) # (B, T', 512)
        T_sub = video.shape[1]

        # áudio (mesmo pipeline da versão framewise)
        B, T_mel, H_mel, W_mel = mel.shape

        mel = mel.reshape(B * T_mel, 1, H_mel, W_mel)
        mel = F.interpolate(mel, size=(128, 1024), mode="bilinear", align_corners=False)
        mel = mel.transpose(2, 3)  # (B * T_mel, 1, 1024, 128)

        with torch.no_grad():
            audio_feat = self.audio_encoder.forward_features(mel)

        audio_feat = audio_feat[:, 0]  # CLS token
        audio_feat = self.audio_projection(audio_feat)  # (B*T_mel, audio_out_dim)
        audio_feat = audio_feat.view(B, T_mel, -1)

        # alinha a sequência de áudio com a sequência de vídeo (T_sub)
        if T_mel == 1 and T_sub > 1:
            audio_feat = audio_feat.expand(-1, T_sub, -1)
        elif T_mel != T_sub:
            # T_mel > 1 e diferente de T_sub: interpola no eixo temporal
            audio_feat = audio_feat.transpose(1, 2)
            audio_feat = F.interpolate(audio_feat, size=T_sub, mode="linear", align_corners=False)
            audio_feat = audio_feat.transpose(1, 2)

        x = torch.cat([video, audio_feat], dim=2)  # concatena vídeo e áudio

        if self.use_fusion:
            x = self.fusion(x)

        # processa sequência temporal
        out, _ = self.lstm(x)          # (B, T_sub, hidden_size * num_directions)
        out = out.max(dim=1).values    # max pooling sobre T_sub

        return self.head(out)  # (B, 1)


'''
Carregar o AudioMAE:

from transformers import AutoModel  # transformers==4.44.0
import einops
import timm
import torchaudio  # torchaudio==2.5.1

model_ae = AutoModel.from_pretrained("hance-ai/audiomae", trust_remote_code=True).to(device)

Observações sobre o vídeo de entrada:
- formato esperado: (B, T, C, H, W), com C = in_channels (1 para grayscale).
- a R2Plus1D-18 pré-treinada espera clipes relativamente curtos (o paper usa
  16 frames). Clipes muito longos aumentam bastante o uso de VRAM, já que a
  conv é 3D — considere usar frame_step > 1 e/ou use_grad_checkpoint=True
  se faltar memória.
'''
