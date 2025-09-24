import torch
import torch.nn as nn

class TinyTransformerClassifier(nn.Module):
    """A minimal Transformer encoder classifier for text.
    Assumes input as (batch, seq_len) token ids, with an embedding layer.
    """
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, num_classes: int = 2, max_len: int = 512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T)
        h = self.emb(x)  # (B, T, D)
        h = self.encoder(h)  # (B, T, D)
        h = h.mean(dim=1)  # simple average pooling
        return self.cls(h)
