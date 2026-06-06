import torch
from gpt import GPT

# ---------------------------------
context = "oh romeo!"
tokens = 1000
temp = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

assert len(context) > 0, "bro put some context or sum"
# ---------------------------------

# create a mapping from characters to integers
stoi = {'\n': 0, ' ': 1, '!': 2, '$': 3, '&': 4, "'": 5, ',': 6, '-': 7, '.': 8, '3': 9, ':': 10, ';': 11, '?': 12, 'A': 13, 'B': 14, 'C': 15, 'D': 16, 'E': 17, 'F': 18, 'G': 19, 'H': 20, 'I': 21, 'J': 22, 'K': 23, 'L': 24, 'M': 25, 'N': 26, 'O': 27, 'P': 28, 'Q': 29, 'R': 30, 'S': 31, 'T': 32, 'U': 33, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38, 'a': 39, 'b': 40, 'c': 41, 'd': 42, 'e': 43, 'f': 44, 'g': 45, 'h': 46, 'i': 47, 'j': 48, 'k': 49, 'l': 50, 'm': 51, 'n': 52, 'o': 53, 'p': 54, 'q': 55, 'r': 56, 's': 57, 't': 58, 'u': 59, 'v': 60, 'w': 61, 'x': 62, 'y': 63, 'z': 64}
itos = { i:ch for i,ch in enumerate(stoi) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
# ---------------------------------

model = GPT()
model.to(device)
model.load_state_dict(state_dict=torch.load("em16hs40la10h4.pt", weights_only=True))

model.eval()
# cont = torch.ones((1,1), dtype=torch.long, device=device)
context = torch.tensor([encode(context)], dtype=torch.long, device=device)
out = decode(model.generate(context, max_tokens=tokens, temp=temp)[0].tolist())