import torch
import torch.nn as nn
import random
from torch.nn import functional as F


# hyperparameters
block_size = 16
batch_size = 64
epochs = 100
max_iters =  epochs * (205411 // batch_size) # len(Xtr) 205411 batches for 1 epoch
eval_interval = max_iters // 5
learning_rate = 2e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
emb_size = 32
n_hidden = 64
n_flattens = block_size.bit_length() - 1  # n.bit_length() - 1 = log2(n)
# dropout = 0.2
# ------------
random.seed(42)

words = open('../data/names.txt', 'r').read().splitlines()
random.shuffle(words)
chars = sorted(list(set(''.join(words))))
stoi = { ch:i+1 for i,ch in enumerate(chars)}
stoi['.'] = 0
itos = { i:ch for ch,i in stoi.items()}
vocab_size = len(itos)

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


# build dataset
def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        # sliding window
        for ch in w + '.':
            ix = stoi[ch] # encode(ch)
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    X, Y = X.to(device), Y.to(device)
    return X, Y

n1 = int(0.9*len(words))

X, Y = build_dataset(words)
Xtr, Ytr = build_dataset(words[:n1])
Xval, Yval = build_dataset(words[n1:])

def get_batch(split):
    dataX = Xtr if split == 'train' else Xval
    dataY = Ytr if split == 'train' else Yval

    ix = torch.randint(low=0, high=dataX.shape[0], size=(batch_size, ))
    x = torch.stack([dataX[i] for i in ix])
    y = torch.stack([dataY[i] for i in ix])
    return x, y

# layers

class Linear(nn.Module):
  def __init__(self, fan_in, fan_out, bias=True):
    super().__init__()
    self.weight = nn.Parameter(torch.randn((fan_in, fan_out)) / fan_in**0.5) # note: kaiming init
    self.bias = nn.Parameter(torch.zeros(fan_out)) if bias else None
  
  def forward(self, x):
    out = x @ self.weight
    if self.bias is not None:
      out += self.bias
    return out

class Flatten(nn.Module):
    def __init__(self, factor=1):
        super().__init__()
        self.factor = factor
        
    def forward(self, idx):
        B, T, C = idx.shape
        idx = idx.view(B, T//self.factor, C*self.factor)
        # if idx.shape[1] == 1:
        #     idx = idx.squeeze(dim=1)
        return idx

class DilatedLayer(nn.Module):
    def __init__(self, fan_in, factor):
        super().__init__()
        self.squash = nn.Sequential(
            Flatten(factor),
            Linear(factor * fan_in, n_hidden, bias=False),
        )
        self.bn = nn.BatchNorm1d(n_hidden)
        self.act = nn.Tanh()

    def forward(self, idx):
        x = self.squash(idx)
        B, T, C = x.shape

        x = x.view(B*T, C) # B, T, C -> B*T, C
        x = self.bn(x)
        x = x.view(B, T, C)
        return self.act(x)
    
class DilatedCasualConvolutions(nn.Module):
    def __init__(self, n_flattens=1):
        super().__init__()
        factor = int(block_size **  (1/n_flattens))
        self.flattens = nn.Sequential(
            DilatedLayer(emb_size, factor), *[DilatedLayer(n_hidden, factor) for _ in range(n_flattens-1)]
        )

    def forward(self, idx):
        return self.flattens(idx)

class WaveNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = nn.Embedding(block_size, emb_size)
        self.dilated = DilatedCasualConvolutions(n_flattens)
        self.unembed = Linear(n_hidden, vocab_size)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))

        x = tok_emb + pos_emb
        # x = x.view(x.shape[0], -1) 
        x = self.dilated(x)
        x = x.squeeze(dim=1)
        logits = self.unembed(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx=None, temp=1, max_new_tokens=100):
        if idx == None:
            idx = torch.zeros((1, block_size), dtype=int, device=device)

        out = []
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs / temp, num_samples=1).item()

            idx = torch.cat((idx[:, 1:], torch.tensor([[ix]], device=device)), dim=1)

            out.append(decode([ix]))
            if ix == 0:
                break
        return out

# training loop
if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    model = WaveNet()
    model.to(device=device)
    model.load_state_dict(torch.load("bnb16e32h64.pt", weights_only=True))
    print(sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 7:
        model = torch.compile(model)
    else:
        print("Triton only supports devices of CUDA Capability >= 7.0")

    for iter in range(max_iters):
        Xb, Yb = get_batch('train')
        logits, loss = model(Xb, Yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % eval_interval == 0 or iter == max_iters - 1:
            print(f"step {iter}, train loss {loss.item():.4f}")

    torch.save(model.state_dict(), "bnb16e32h64.pt")
    model.eval()
    for _ in range(5):
        print("".join(model.generate()))

    # loss eval
    @torch.no_grad()
    def split_loss(split):
        x, y = {
            'train': (Xtr, Ytr),
            'val': (Xval, Yval)
        }[split]
        _, loss = model(x, y)
        print(split, loss.item())

    split_loss('train')
    split_loss('val')
