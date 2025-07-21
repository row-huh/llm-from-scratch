import re
from Simpletokenizer import SimpleTokenizerV1

with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()


# preprocessing before assiginin token ids
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(set(preprocessed))

vocab = {token:integer for integer, token in enumerate(all_words)}
print("Pre tokenization: ", vocab)

tokenizer = SimpleTokenizerV1(vocab=vocab)
text = """
    "It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride
"""

ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))