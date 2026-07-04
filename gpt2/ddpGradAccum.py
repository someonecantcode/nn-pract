import torch
import torch.nn as nn
import tiktoken
import os
import torch.distributed as dist
from torch.nn import functional as F
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class GPTConfig:
    batch_size: int = 32
    block_size: int = 256
    vocab_size: int = 50304 # tokenizer.n_vocab # 50257, 50000 merges + 256 byte + <endoftext>
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    lr: float = 3e-4

# -------------------------------------------------------------------------------

ddp = int(os.environ.get('RANK', -1)) != -1 # using ddp torch run?
if ddp:
    assert torch.cuda.is_available(), "you need CUDA for DDP"
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # which gpu 0,1,2,..
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # multinode
    ddp_world_size = int(os.environ['WORLD_SIZE']) # total gpus
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device=device)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(device)

master_process = ddp_rank == 0
print(ddp)
print(ddp_rank)
print(ddp_local_rank)
print(ddp_world_size)

# -------------------------------------------------------------------------------

tokenizer = tiktoken.get_encoding('gpt2')
class DataLoader():
    def __init__(self, B, T, gpu_rank, total_gpus):
        self.B = B
        self.T = T
        self.gpu_rank = gpu_rank
        self.total_gpus = ddp_world_size

        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9*len(data)) # first 90% will be train, rest val
        self.train_data = data[:n]
        self.val_data = data[n:]

        self.current_pos = self.B * self.B * ddp_rank
        if master_process:
            print(f"tokens: {len(self.train_data)}")
            print(f"1 epoch is {len(self.train_data) // (self.B * self.T * ddp_world_size)} iters")

    def get_batch(self, split): # outdated(still works) random batch from gpt1
        batch_size, block_size = self.B, self.T
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(low=0, high=len(data)-block_size, size=(batch_size, ))

        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
    
    def next_batch(self, split):
        B, T = self.B, self.T
        data = self.train_data if split == 'train' else self.val_data
        buffer = data[self.current_pos : self.current_pos + B*T + 1]
        
        x = buffer[:-1].view(B, T)
        y = buffer[1:].view(B, T)
        x, y = x.to(device), y.to(device)

        self.current_pos += B * T * self.total_gpus
        if self.current_pos >= (len(data) - (1+B*T)):
            self.current_pos = self.B * self.B * ddp_rank
        return x, y
# -------------------LAYERS-------------------------------------------------------

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

# -------------------LAYERS END-------------------------------------------------------

import inspect
import time

eval_iter = 10
max_iters = 2
tokens_per_step = 524288
grad_accum_steps = tokens_per_step // (GPTConfig().batch_size * GPTConfig().block_size)
# grad_accum_steps = 16


loader = DataLoader(B=GPTConfig.batch_size, T=GPTConfig().block_size, gpu_rank=ddp_rank, total_gpus=ddp_world_size) # GPTConfig.block_size
loader.next_batch('train')

model = GPT(GPTConfig(vocab_size=50304))
model = model.to(device=device)

if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 7:
    model = torch.compile(model)
else:
    print("Triton only supports devices of CUDA Capability >= 7.0")

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) # all reduce and deposit gradients


fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
used_fused = fused_available and 'cuda' in device
optimizer = torch.optim.AdamW(model.parameters(), lr=model.config.lr, fused=used_fused)

# training loop
for i in range(1, max_iters): 
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        Xb, Yb = loader.next_batch('train')
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # slower on pascal arch
            logits, loss = model(Xb, Yb)

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        if ddp: # wait and sync with all gpus at the end
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward() # changed due to ddp wrap
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    optimizer.step() # since all gpu's have the same gradients, we are "sync" with the weights after we update as well
    
    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    torch.cuda.synchronize()
    t1 = time.time()

    if master_process:
        dt = (t1 - t0) 
        tokens = (loader.B * loader.T * grad_accum_steps * ddp_world_size) / dt
        print(f"step: {i}, loss: {loss_accum.item():.4f}, norm: {norm:.2f}, time: {1000 * dt:.2f} ms, tok/sec: {tokens:.2f}")
        t0 = time.time()

if ddp:
    dist.destroy_process_group()