from transformers import GPT2LMHeadModel, GPT2Tokenizer
#pip install transformers
#pip install torch
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
print(tokenizer.eos_token_id)
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)

sentence = "Video games are addicting"
input_ids = tokenizer.encode(sentence, return_tensors="pt")
print(input_ids)

output = model.generate(input_ids, max_length = 100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
