from DummyGPTModel import FeedForward, GPT_CONFIG_124M
import torch

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape) 