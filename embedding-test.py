import torch

# # assuming token ids of the input
# input_ids = torch.tensor([2, 3, 5, 1])


# # total size of the vocab i.e words in training data
# vocab_size = 6

# # 3 dimensions of vector (programmatically, this is an array with 3 items)
# output_dim = 3

# # instantiating an embedding layer where you defined the rows/columsn
# # in a sense that vocab_size is the rows in the layer (i think that's nodes in a neural network)
# # and output_dim is the number of items in that node ?
# torch.manual_seed(123)
# embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# print(embedding_layer.weight)

# print(embedding_layer(input_ids))



from util import create_dataloader_v1
# embedding with positional embedding appraoch

#fetch text
with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

print(len(raw_text))

vocab_size = 50256
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# embed into 256 dimensional vectors
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)



# absolute embedding approach
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

# add these directly to the token embeddings, where PyTorch will add
# the 4 × 256–dimensional pos_embeddings tensor to each 4 × 256–dimensional token
# embedding tensor in each of the eight batches:

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)