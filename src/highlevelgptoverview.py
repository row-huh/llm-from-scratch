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
