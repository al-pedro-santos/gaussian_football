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
        audio_chunk_size: int = 16,  # processa o AudioMAE em blocos de até N
                                      # espectrogramas por vez, em vez de
                                      # B*T_mel de uma só passada — evita pico
                                      # de memória quando o grupo (B) é grande
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
                        # média dos pesos dos 3 canais RGB -> 1 canal grayscale
                        # (usar .sum() aqui triplicava a magnitude das ativações
                        # do primeiro conv em relação ao que a rede pré-treinada
                        # espera, deslocando a estatística de ativação logo na
                        # entrada; .mean() preserva a escala esperada pelos
                        # pesos KINETICS400_V1)
                        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                    else:
                        c = min(in_channels, 3)
                        new_conv.weight[:, :c].copy_(old_conv.weight[:, :c])
            self.cnn[0][0] = new_conv

        # Normalização de entrada compatível com os pesos KINETICS400_V1
        # (mean≈[0.43,0.39,0.38], std≈[0.23,0.22,0.22] por canal RGB, na
        # transformação oficial do torchvision). Sem isso, especialmente com
        # o backbone congelado (freeze_mode="frozen"), os vídeos entram fora
        # da distribuição em que a CNN foi pré-treinada e os features saem
        # quase como ruído — empurrando o modelo pra uma saída praticamente
        # constante. O vídeo é esperado em [0, 1] (ex.: já dividido por 255
        # no dataloader) antes de chegar aqui.
        if pretrained:
            if in_channels == 1:
                # grayscale: usa a média dos 3 canais RGB como aproximação
                # (equivalente ao mesmo critério usado para adaptar os pesos
                # do conv acima)
                kinetics_mean = [(0.43216 + 0.394666 + 0.37645) / 3]
                kinetics_std = [(0.22803 + 0.22145 + 0.216989) / 3]
            elif in_channels == 3:
                kinetics_mean = [0.43216, 0.394666, 0.37645]
                kinetics_std = [0.22803, 0.22145, 0.216989]
            else:
                kinetics_mean = [0.43216, 0.394666, 0.37645][:in_channels]
                kinetics_std = [0.22803, 0.22145, 0.216989][:in_channels]
                while len(kinetics_mean) < in_channels:
                    kinetics_mean.append(kinetics_mean[-1])
                    kinetics_std.append(kinetics_std[-1])
            self.register_buffer(
                "video_mean", torch.tensor(kinetics_mean).view(1, in_channels, 1, 1, 1)
            )
            self.register_buffer(
                "video_std", torch.tensor(kinetics_std).view(1, in_channels, 1, 1, 1)
            )
        else:
            self.video_mean = None
            self.video_std = None

        self.frame_step = frame_step
        self.use_grad_checkpoint = use_grad_checkpoint
        self.audio_chunk_size = max(1, int(audio_chunk_size))

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
        # OBS: a subamostragem temporal (frame_step) já é feita no
        # MultiModalDataset._load_sample (loader_multimodal_frac2.py), antes
        # do vídeo chegar aqui. Aplicá-la de novo aqui subamostraria em
        # frame_step**2, reduzindo a sequência mais do que o esperado sem
        # nenhum ganho de memória real (o tensor já chega menor do disco).
        # Por isso o corte foi removido daqui; self.frame_step é mantido só
        # para compatibilidade com código externo que leia esse atributo.

        # Conv3d espera (B, C, T, H, W)
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        if self.video_mean is not None:
            video = (video - self.video_mean) / self.video_std
        video = self._forward_cnn(video) # (B, 512, T', H', W')
        video = video.mean(dim=[3, 4]) # pooling espacial -> (B, 512, T')
        video = video.transpose(1, 2) # (B, T', 512)
        T_sub = video.shape[1]
        # exposto para diagnóstico externo (ex.: sanity check no notebook) —
        # se T_sub ficar em 1-2, a BiLSTM está processando sequências
        # praticamente sem dimensão temporal, o que compromete o que ela
        # pode aprender; vale revisar FRAME_STEP / duração dos clipes.
        self.last_T_sub = T_sub

        # áudio (mesmo pipeline da versão framewise)
        B, T_mel, H_mel, W_mel = mel.shape

        mel = mel.reshape(B * T_mel, 1, H_mel, W_mel)
        mel = F.interpolate(mel, size=(128, 1024), mode="bilinear", align_corners=False)
        mel = mel.transpose(2, 3)  # (B * T_mel, 1, 1024, 128)

        # Processa o AudioMAE em blocos: sob torch.no_grad() não há grafo pra
        # backward, mas o forward de um transformer com B*T_mel linhas de
        # uma vez só ainda aloca ativações proporcionais a esse total. Se o
        # grupo (B) for grande, isso sozinho pode estourar a VRAM. Processar
        # em chunks limita o pico sem mudar o resultado.
        chunks = []
        with torch.no_grad():
            for start in range(0, mel.shape[0], self.audio_chunk_size):
                chunk = mel[start:start + self.audio_chunk_size]
                chunks.append(self.audio_encoder.forward_features(chunk))
        audio_feat = torch.cat(chunks, dim=0)

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