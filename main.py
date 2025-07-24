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

print(text)

print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))


