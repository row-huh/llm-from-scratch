import re

with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

print("Total number of characters: ", len(raw_text))
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]


all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)


vocab = {token:integer for integer, token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break