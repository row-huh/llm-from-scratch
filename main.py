import re

with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

print("Total number of characters: ", len(raw_text))
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print(len(preprocessed))

print(preprocessed[:30])