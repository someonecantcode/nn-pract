import torch
from wavenet import WaveNet 

block_size = 16
total_samples = 5
prompt = "eile"
assert (len(prompt) <= block_size), "context too large"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------
words = open('../data/names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = { ch:i+1 for i,ch in enumerate(chars)}
stoi['.'] = 0
itos = { i:ch for ch,i in stoi.items()}
vocab_size = len(itos)

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
# ---------------------------------

model = WaveNet()
model.to(device=device)
model.load_state_dict(torch.load("b16e32h64.pt", weights_only=True))

context = [0] * block_size
context = context[:block_size - len(prompt)] + encode(prompt) 
context = torch.tensor([context])
context = context.to(device)
for _ in range(total_samples):
    print(prompt + "".join(model.generate(idx=context)))
