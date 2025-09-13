import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257, # total number of words 
    "context_length": 1024, 
    "embd_dim": 768, 
    "n_heads": 12, 
    "n_layers": 12, 
    "drop_rate": 0.1, 
    "qkv_bias": False,  # to be implemented in chapter 6
}

# nn.module is base clas for neural networks
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        # initialize super class - in this case its nn.module - or a neural network
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embd_dim"])
        self.positional_embedding = nn.Embedding(cfg["vocab_size"], cfg["embd_dim"])
        self.drop_embedding = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["embd_dim"])
        self.out_head = nn.Linear(
            cfg["embd_dim"], cfg["vocab_size"], bias=False
        )
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeddings = self.token_embedding(in_idx)
        positional_embeddings = self.positional_embedding(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = token_embeddings + positional_embeddings
        x = self.drop_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        
        logits = self.out_head(x)
        return logits




# These are placeholders for now, will be used later
class DummyTransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        
    def forward(self, x):
        return x
    

# These are placeholders for now, will be used later
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift