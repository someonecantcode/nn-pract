import torch
from gpt import GPT

# ---------------------------------
context = "oh romeo!"
tokens = 100
temp = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

assert len(context) > 0, "bro put some context or sum"
# ---------------------------------
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
# ---------------------------------

model = GPT()
model.to(device)
model.load_state_dict(state_dict=torch.load("em16hs40la10h4.pt", weights_only=True))

model.eval()
# cont = torch.ones((1,1), dtype=torch.long, device=device)
context = torch.tensor([encode(context)], dtype=torch.long, device=device)
print(decode(model.generate(context, max_tokens=tokens, temp=temp)[0].tolist()))