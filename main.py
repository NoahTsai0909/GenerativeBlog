from transformers import GPT2LMHeadModel, GPT2Tokenizer
#pip install transformers
#pip install torch
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
print(tokenizer.eos_token_id)
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)

sentence = "I like ice cream"
input_ids = tokenizer.encode(sentence, return_tensors="pt")
print(input_ids)
