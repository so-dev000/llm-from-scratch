import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch_device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = AutoModelForCausalLM.from_pretrained(
    "gpt2", pad_token_id=tokenizer.eos_token_id
).to(torch_device)

# Greedy search

inputs = tokenizer("I enjoy walking with my cute dog", return_tensors="pt").to(
    torch_device
)
greedy_output = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
