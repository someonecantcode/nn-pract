import torch
import torch.nn as nn
from torch.nn import functional as F

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )
    
    def forward(self, x):
        return x + self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.multiattn = nn.MultiheadAttention(embed_dim=config.n_embd, num_heads=config.n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FFN(config)
    
    def forward(self, x):
        device = x.device
        norm = self.ln1(x)
        attn, _ = self.multiattn(query=norm, key=norm, value=norm, need_weights=False, attn_mask=torch.triu(torch.ones(x.shape[-2], x.shape[-2], device=device, dtype=torch.bool), diagonal=1))
        x = x + attn
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln = nn.LayerNorm(config.n_embd)
        ))

        self.proj = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.proj.weight

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
        device = idx.device
        assert T <= self.config.block_size, f"context length of {T} exceeds block_size of {self.config.block_size}"
        
        pos_emb = self.transformer.wpe(torch.arange(T, dtype=torch.long, device=device)) # (block, embed)
        tok_emb = self.transformer.wte(idx) # (B,T,C) -> (batch, block, embed)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln(x)

        if targets is None:
            loss = None
            logits = x[:, [-1], :] # only need to calc last ones for generation
            logits = self.proj(logits)
        else:
            logits = self.proj(x)
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)    
    
        return logits, loss


    def generate(self, idx, max_tokens=1, temp=1):
        for _ in range(max_tokens):
            logits, _ = self(idx[:,-self.config.block_size:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs / temp, num_samples=1)
            idx = torch.cat((idx, ix), dim=1) # lol no need to cut
            # print(decode(ix[0].tolist()), end="", flush=True)
        return idx