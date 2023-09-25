from transformers import GPT2LMHeadModel, GPT2Tokenizer
#pip install transformers
#pip install torch
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
print(tokenizer.eos_token_id)
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)

sentence = input('Please enter the first sentence of the blog post as a prompt: \n')
maxlen = input('Please enter the max word count for the blog post:\n')
input_ids = tokenizer.encode(sentence, return_tensors="pt")

output = model.generate(input_ids, max_length = int(maxlen), num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
