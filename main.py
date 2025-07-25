import re
from util import SimpleTokenizerV1, SimpleTokenizerV2, create_dataloader_v1
import tiktoken
from torch.utils.data import DataLoader
import torch


with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()


# preprocessing before assiginin token ids
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(list(set(preprocessed)))
all_words.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer, token in enumerate(all_words)}


tokenizer = SimpleTokenizerV2(vocab=vocab)
text = """
    "It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride
"""

ids = tokenizer.encode(text)


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
second_batch = next(data_iter)
#print("First batch tensor:", first_batch)
#print("Second batch tensor:", second_batch)


# testing embeddings
input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#print(embedding_layer.weight)

# apply to a token id 3
#print(embedding_layer(torch.tensor([3])))

# embedding layer of all input ids
print(embedding_layer(input_ids))