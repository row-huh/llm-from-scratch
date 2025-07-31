# read the-verdict
# tokenize (using the regex approach)
# sort all and convert into set to remove duplicates
# enumerate each item and store in a dict object (call it vocabulary)
# introduce <| unk |> and <| end of text|> special tokens into your vocabulary as well

# create a class that takes in text, creates a vocab, tokenizes other text
# based on that vocab and if any word doesnt exist, it will use a special token


# this tokenizer that will be created has a much simpler approach
# it's better to go with tiktoken and use bpe algorithm

import tiktoken
tokenizer = tiktoken.get_encoding("gpt-2") # fetch vocab from gpt2 i think?

text = (
"Hello, do you like tea? <|endoftext|> In the sunlit terraces"
"of someunknownPlace."
)

# convert into token ids
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)


# convert back into original tokens
strings = tokenizer.decode(integers)
print(strings)


# create a datasetloader class responsible for creating
# input tensors and target tensors (see gptdatasetv1)

# then finally create an instantiator/loader function that will instantiate
# gptdatasetloader for you and instantiate pytorch's dataloader as well

# now you can pass in your text + some parameters into this function and get
# back an efficient dataloader - data-iter = iter(dataloader) to convert into iterative objeect
# then next(data_iter) gets you batches one by one (batches means input/target pairs tensors)


# finally convert into embedding layer using torch.nn.Embedding(vocab_size, output_dim)
