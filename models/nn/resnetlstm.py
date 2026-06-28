import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class ResNetLSTM(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        frame_step: int = 2,
        hidden_size: int = 256,
        num_layers: int = 1,
        use_dropout: bool = False,
        dropout_p: float = 0.3,
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

        self.frame_step = frame_step # vê um frame a cada frame_step frames

        # arquitetura da ResNet disponível em https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        
        # lista todos os módulos da ResNet e remove o classificador no final, 
        # deixando o AdaptiveAvgPool: (B*T, cnn_out_size, 1, 1)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        # substitui primeiro conv para aceitar grayscale (1 canal), antes era 3 na entrada
        self.cnn[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # BiLSTM processa a sequência de features por frame
        self.lstm = nn.LSTM(
            cnn_out_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_p if num_layers > 1 else 0,
        )

        # cabeça de regressão: max pool -> 128 -> 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p) if use_dropout else nn.Identity(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # C vai ser 1 sempre (grayscale)


        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # subsample temporal
        x = x[:, ::self.frame_step, :, :, :]
        T_sub = x.shape[1]

        # extrai features por frame com a CNN
        x = x.reshape(B * T_sub, C, H, W)
        x = self.cnn(x)           # (B*T_sub, cnn_out_size, 1, 1)
        x = x.view(B, T_sub, -1)  # (B, T_sub, cnn_out_size)

        # processa sequência temporal
        out, _ = self.lstm(x)          # (B, T_sub, hidden_size*2)
        out = out.max(dim=1).values    # (B, hidden_size*2), max pooling sobre T

        return self.head(out)  # (B, 1)
    
class ResNet_Cut_LSTM(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        frame_step: int = 2,
        hidden_size: int = 256,
        num_layers: int = 1,
        use_dropout: bool = False,
        dropout_p: float = 0.3,
        cut_layer: int = 4,  # Ex: 4 pula o conv1/bn1/relu/maxpool e vai direto pro layer1
        LSTM_bidirectional : bool = True
    ):
        super().__init__()

        backbones = {
            "resnet18": (resnet18, ResNet18_Weights.DEFAULT, 512),
            "resnet34": (resnet34, ResNet34_Weights.DEFAULT, 512),
            "resnet50": (resnet50, ResNet50_Weights.DEFAULT, 2048),
        }

        assert backbone_name in backbones, f"backbone_name deve ser um de {list(backbones.keys())}"
        model_fn, weights, cnn_out_size = backbones[backbone_name]
        backbone = model_fn(weights=weights)

        self.frame_step = frame_step 
        modules_list = list(backbone.children())
        
        # Garante que o corte não vai apagar a rede inteira (deve sobrar pelo menos o AvgPool no idx 8)
        assert cut_layer < len(modules_list) - 1, f"cut_layer máximo permitido para {backbone_name} é {len(modules_list) - 2}"

        # Se cut_layer == 0, usamos o comportamento padrão (substituir conv1)
        if cut_layer == 0:
            self.input_adapter = nn.Identity()
            self.cnn = nn.Sequential(*modules_list[:-1])
            self.cnn[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Se cut_layer > 0, criamos a ponte para o corte
        else:
            modules = modules_list[cut_layer:-1]
            first_remaining_layer = modules[0]
            
            # Descobre dinamicamente quantos canais a camada sobrevivente espera
            if isinstance(first_remaining_layer, nn.Sequential) and hasattr(first_remaining_layer[0], 'conv1'):
                expected_channels = first_remaining_layer[0].conv1.in_channels
            elif hasattr(first_remaining_layer, 'in_channels'):
                expected_channels = first_remaining_layer.in_channels
            else:
                expected_channels = 64 # Fallback padrão para os primeiros blocos
            
            # Cria o adaptador: Transforma 1 canal nos canais esperados.
            # Nota: Usamos stride=2 para ajudar na redução de resolução que as camadas cortadas fariam.
            self.input_adapter = nn.Conv2d(1, expected_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.cnn = nn.Sequential(*modules)
            
            print(f"[Config] ResNet cortada no bloco {cut_layer}. Adaptador criado: 1 -> {expected_channels} canais.")

        # BiLSTM processa a sequência de features por frame
        self.lstm = nn.LSTM(
            cnn_out_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=LSTM_bidirectional,
            dropout=dropout_p if num_layers > 1 else 0,
        )

        # cabeça de regressão
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p) if use_dropout else nn.Identity(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        # subsample temporal
        x = x[:, ::self.frame_step, :, :, :]
        T_sub = x.shape[1]

        # extrai features por frame
        x = x.reshape(B * T_sub, C, H, W)
        
        # O adaptador entra em ação aqui antes da ResNet cortada!
        x = self.input_adapter(x) 
        x = self.cnn(x)    
        
        x = x.view(B, T_sub, -1)  

        # processa sequência temporal
        out, _ = self.lstm(x)          
        out = out.max(dim=1).values    

        return self.head(out)