import tiktoken
import torch
from DummyGPTModel import *

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
# i did update the text but each batch must be 
# equal size for the torch.stack thing to work
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0)

print(batch)


torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch) # same as model.forward(batch). Why?? because yes
print("Output shape:", logits.shape)
print(logits)

# example of figure 4.5 (page 100)
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(type(out))

# calculating mean of this small neural network and variance it - they will later be normalized
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var)