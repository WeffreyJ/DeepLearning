import torch.nn as nn
class TinyTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)
    def forward(self,x):
        h = self.enc(self.emb(x))
        return self.cls(h.mean(1))
