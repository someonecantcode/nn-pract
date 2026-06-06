import torch
import torch.nn as nn
import einops
from torch.nn import functional as F


# hyperparams
block_size = 8
batch_size = 128

n_embd = 16
head_size = 40
n_layer = 10 # blocks
n_heads = 4

# 1115394 total examples so 1115394 // batchsize = # of iters to get an epoch
epoch = 10
max_iters = epoch * (1115394 // batch_size)
eval_interval = 5
lr = 8e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# ------------

with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(low=0, high=len(data)-block_size, size=(batch_size, ))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Layers -----------------------------------
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
        )
    
    def forward(self, x):
        return x + self.net(x)
    
def ScaledDotProductAttention(q, k, v):
    d_k = q.shape[-1]

    wei = q @ k.transpose(-1, -2) * d_k**(-0.5) 

    tril = torch.tril(torch.ones((k.shape[-2], k.shape[-2]), device=device))
    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    out = wei @ v

    return out
    
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # self.headtoembd = nn.Linear(head_size, n_embd)
    
    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x) 
        k = self.key(x)   # (B,T,hs)
        v = self.value(x)

        out = ScaledDotProductAttention(q, k ,v)
        # out = self.headtoembd(out)
        return out # do residual at the blocks instead
    
class batchedMultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_weights = nn.Linear(n_embd, 3 * head_size, bias=False)
        self.proj = nn.Linear(head_size, n_embd)

    def forward(self, x):
        B, T, C = x.shape

        qkv =  self.qkv_weights(x) # B, T, 3 * head_size
        q, k, v = tuple(einops.rearrange(qkv, "b t (k nh hs) -> k b nh t hs", k=3, nh=n_heads)) # tuple to remove the 0th dim

        out = ScaledDotProductAttention(q, k ,v)

        # this thing sucks smh
        # out = out.transpose(1, 2).contiguous().view(B, T, head_size) # i forgot u need to switch nh, T smh smh -> also contigous is needed for .transopse
        out = einops.rearrange(out, 'b nh t hs -> b t (nh hs)')
        out = self.proj(out)

        return out
    
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.multiattn = batchedMultiHeadAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FFN()
    
    def forward(self, x):
        x = x + self.multiattn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ffdw = FFN()
        self.unembed = nn.Linear(n_embd, vocab_size)

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
        
        tok_emb = self.tok_emb(idx) # (B,T,C) -> (batch, block, embed)
        pos_emb = self.pos_emb(torch.arange(T, device=device)) # (block, embed)
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ffdw(x)
        
        
        if targets is None:
            loss = None
            logits = x[:, [-1], :] # only need to calc last ones for generation
            logits = self.unembed(logits)
        else:
            logits = self.unembed(x)
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)    
    
        return logits, loss


    def generate(self, idx, max_tokens=1, temp=1):
        for _ in range(max_tokens):
            logits, _ = self(idx[:,-block_size:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs / temp, num_samples=1)
            idx = torch.cat((idx, ix), dim=1) # lol no need to cut
            print(decode(ix[0].tolist()), end="", flush=True)
        return idx
# -----------------------------------

if __name__ == "__main__":
    model = GPT()
    model.to(device)

    model.load_state_dict(state_dict=torch.load("em16hs40la10h4.pt", weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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

        if iter % (max_iters // eval_interval) == 0 or iter == max_iters - 1:
            print(loss.item())
    torch.save(getattr(model, "_orig_mod", model).state_dict(), "em16hs40la10h4.pt")



    model.eval()
    n_iter_eval = len(val_data) // batch_size #  around (2.7 seconds per 100 n_iter_eval)

    @torch.no_grad()
    def get_total_loss():
        out = {}
        for split in ['train', 'val']:
            data = train_data if split == 'train' else val_data
            losses = torch.zeros(n_iter_eval) # calculating loss over entire epoch over batches
            
            for k in range(n_iter_eval):
                ix = torch.arange(batch_size)
                x = torch.stack([data[k+i:k+i+block_size] for i in ix])
                y = torch.stack([data[k+i+1:k+i+block_size+1] for i in ix])
                x, y = x.to(device), y.to(device)

                _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        return out

    losses = get_total_loss()
    print(f"train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    cont = torch.ones((1,1), dtype=torch.long, device=device)
    print(decode(model.generate(cont, 100)[0].tolist()))