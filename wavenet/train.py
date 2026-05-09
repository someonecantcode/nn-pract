import torch
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()
print(words[:5])
print(len(words))

# encode chars to integers
chars = sorted(list(set(''.join(words))))
stoi = { ch:i+1 for i,ch in enumerate(chars)}
stoi['.'] = 0
itos = { i:ch for ch,i in stoi.items()}
vocab_size = len(itos)
print(itos)

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hello"))
print(decode([8, 9, 0]))