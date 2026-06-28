import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioMAE_HighlightClassifier(nn.Module):
    """
    Modelo de áudio puro: recebe o mel-spectrogram de um clipe e prevê se
    aquele trecho contém um highlight de jogo de futebol (classificação binária).

    Ideia: extrair embeddings por "frame" de áudio com o AudioMAE (congelado),
    agregar a sequência temporal (LSTM ou pooling simples) e jogar numa cabeça
    de classificação com 1 logit (BCEWithLogitsLoss).

    `frame_step` controla o subsampling temporal dos frames de mel antes de
    passar pelo AudioMAE — útil pra reduzir custo computacional e/ou alinhar
    com o frame_step usado no modelo de vídeo (ex.: se o vídeo usa
    frame_step=2 numa janela de N segundos, pode fazer sentido o áudio usar
    uma taxa equivalente, dependendo de como T foi construído no seu dataset).

    Depois, esse modelo pode ser combinado com um modelo de vídeo de duas formas:
      1) Fusão tardia (late fusion): usar a probabilidade (sigmoid do logit) ou
         o próprio logit como uma feature extra concatenada às features de vídeo.
      2) Fusão de features: usar o embedding agregado (saída antes do `head`,
         por exemplo `audio_embedding`) como vetor de contexto de áudio para o
         modelo de vídeo, igual ao fusion_dim que você já usa no
         ResNetLSTM_MultiModal.

    Todos os clipes passados devem ter menos do que 10 segundos, senão o AudioMae corta
    eles.

    Só funciona com a loss de BCE

    Usamos:
    AutoModel.from_pretrained("hance-ai/audiomae", trust_remote_code=True).to(device)

    """

    def __init__(
        self,
        audiomae,
        audio_out_dim: int = 512,
        frame_step: int = 1,
        hidden_size: int = 256,
        num_layers: int = 1,
        use_temporal_model: bool = True,
        use_dropout: bool = True,
        dropout_p: float = 0.3,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.frame_step = frame_step  # vê um frame de áudio a cada frame_step frames

        self.audio_encoder = audiomae.encoder

        if freeze_encoder:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
            self.audio_encoder.eval()

        self.freeze_encoder = freeze_encoder

        # projeta o embedding do AudioMAE (768) para um tamanho menor
        self.audio_projection = nn.Linear(768, audio_out_dim)

        self.use_temporal_model = use_temporal_model

        if use_temporal_model:
            # BiLSTM agrega a sequência de embeddings por frame de áudio
            self.lstm = nn.LSTM(
                audio_out_dim,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_p if num_layers > 1 else 0,
            )
            head_in_dim = hidden_size * 2
        else:
            # sem LSTM: só faz pooling (max) direto sobre os embeddings
            head_in_dim = audio_out_dim

        self.embedding_dim = head_in_dim  # útil pra fusão posterior com o modelo de vídeo

        # cabeça de classificação binária: embedding -> 128 -> 1 logit
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p) if use_dropout else nn.Identity(),
            nn.Linear(128, 1),
        )

    def encode(self, mel):
        """
        Extrai o embedding agregado de áudio (antes da cabeça de classificação).
        Retorna (B, embedding_dim). Útil para fundir depois com features de vídeo.

        mel: (B, T, H_mel, W_mel) — T = número de "frames"/janelas de áudio do clipe.
        """
        # subsample temporal, igual ao frame_step usado no vídeo
        mel = mel[:, ::self.frame_step]

        B, T, H_mel, W_mel = mel.shape

        mel = mel.reshape(B * T, 1, H_mel, W_mel)
        mel = F.interpolate(mel, size=(128, 1024), mode="bilinear", align_corners=False)
        mel = mel.transpose(2, 3)  # (B*T, 1, 1024, 128), mesmo formato esperado pelo AudioMAE

        if self.freeze_encoder:
            with torch.no_grad():
                audio_feat = self.audio_encoder.forward_features(mel)
        else:
            audio_feat = self.audio_encoder.forward_features(mel)

        audio_feat = audio_feat[:, 0]  # CLS token
        audio_feat = self.audio_projection(audio_feat)  # (B*T, audio_out_dim)
        audio_feat = audio_feat.view(B, T, -1)           # (B, T, audio_out_dim)

        if self.use_temporal_model:
            out, _ = self.lstm(audio_feat)       # (B, T, hidden_size*2)
            embedding = out.max(dim=1).values     # max pooling sobre T
        else:
            embedding = audio_feat.max(dim=1).values  # (B, audio_out_dim)

        return embedding

    def forward(self, mel):
        """
        mel: (B, T, H_mel, W_mel)
        retorna logit (B, 1) -> use BCEWithLogitsLoss durante o treino,
        e torch.sigmoid(logit) na inferência para obter a probabilidade de highlight.
        """
        embedding = self.encode(mel)
        logit = self.head(embedding)
        return logit


if __name__ == "__main__":
    # exemplo de uso / sanity check com tensores aleatórios
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_ae = AutoModel.from_pretrained("hance-ai/audiomae", trust_remote_code=True).to(device)

    model = AudioMAE_HighlightClassifier(audiomae=model_ae, frame_step=2).to(device)

    B, T, H_mel, W_mel = 2, 8, 128, 128  # T maior aqui pra sobrar frames após o frame_step
    mel = torch.randn(B, T, H_mel, W_mel, device=device)

    logits = model(mel)
    probs = torch.sigmoid(logits)

    print("logits:", logits.shape)   # (B, 1)
    print("probs :", probs)

    # exemplo: pegar embedding pra fusão posterior com modelo de vídeo
    emb = model.encode(mel)
    print("embedding:", emb.shape)   # (B, model.embedding_dim)