import torch
from transformers import AutoModel
from models.nn.resnetlstm_multimodal import ResNetLSTM_MultiModal

device = "cuda" if torch.cuda.is_available() else "cpu"

model_ae = AutoModel.from_pretrained("hance-ai/audiomae", trust_remote_code=True).to(device)

B = 2
video = torch.randn(B, 50, 1, 224, 224, device=device)
mel = torch.randn(B, 1, 128, 1024, device=device)

for use_fusion in [True, False]:
    torch.cuda.reset_peak_memory_stats(device)

    model = ResNetLSTM_MultiModal(audiomae=model_ae, backbone_name="resnet18", use_fusion=use_fusion).to(device)
    out = model(video, mel)

    loss = out.sum()
    loss.backward()

    print(f"use_fusion={use_fusion}  out.shape={out.shape}")
    print(f"  peak mem: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")

    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            print("  sem gradiente:", name)

    del model
    torch.cuda.empty_cache()