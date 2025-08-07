from utility import create_dataloader_v1
import tiktoken
from torch.utils.data import DataLoader
import torch

# read training text
with open("../the-verdict.txt", 'r') as file:
    raw_text = file.read()


# creating a new neural network layer
vocab_size = 50257
output_dim = 8
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# create a datasetloader class responsible for creating
# input tensors and target tensors (see gptdatasetv1)

dataloader = create_dataloader_v1(raw_text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0)

# fetch out all data split into batches
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
token_embeddings = token_embedding_layer(inputs)

# finally convert into embedding layer using torch.nn.Embedding(vocab_size, output_dim)
context_length = 256
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)


input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)