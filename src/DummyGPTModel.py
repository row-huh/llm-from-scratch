import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257, # total number of words 
    "context_length": 1024, 
    "embd_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "dropout": 0.1,
    "qkv_bias": False,  # to be implemented in chapter 6
}

# nn.module is base clas for neural networks
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        # initialize super class - in this case its nn.module - or a neural network
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embd_dim"])
        self.positional_embedding = nn.Embedding(cfg["vocab_size"], cfg["embd_dim"])
