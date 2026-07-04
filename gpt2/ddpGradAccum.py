import torch
import os
import inspect
import time
import torch.distributed as dist
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt import GPT
from DataLoader import DataLoader

# torchrun --standalone --nproc-per-node=8 file.py

os.environ["TORCHINDUCTOR_CACHE_DIR"] = ".inductor_cache"
start_time = time.time()
time_limit = 3600

@dataclass
class GPTConfig:
    batch_size: int = 32
    block_size: int = 256
    vocab_size: int = 50257 # tokenizer.n_vocab # 50257, 50000 merges + 256 byte + <endoftext>
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
if master_process:
    print(ddp_world_size)

# -------------------------------------------------------------------------------
ckpt_path = "ddp_4.pt"
def load_checkpoint(ckpt_path, model):
    if not (os.path.exists(ckpt_path)):
        return 1
    
    checkpoint = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(checkpoint['model'] )

    return checkpoint['step']

def save_checkpoint(raw_model, step):
    checkpoint = {
        'model': raw_model.state_dict(),
        'step': step
    }
    torch.save(checkpoint, f"ddp_{step}.pt")
# -------------------------------------------------------------------------------
max_iters = 3
tokens_per_step = 524288
grad_accum_steps = tokens_per_step // (GPTConfig().batch_size * GPTConfig().block_size * ddp_world_size)
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
raw_model = model.module if ddp else model


fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
used_fused = fused_available and 'cuda' in device
optimizer = torch.optim.AdamW(model.parameters(), lr=raw_model.config.lr, fused=used_fused)

left_off_step = load_checkpoint(ckpt_path=ckpt_path, model=model)

# training loop 
for step in range(left_off_step, max_iters+1): 

    elapsed_time = time.time() - start_time
    if elapsed_time <= time_limit:

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            Xb, Yb = loader.next_batch('train')
            Xb, Yb = Xb.to(device), Yb.to(device)
            
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
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.cuda.synchronize()
        t1 = time.time()

        if master_process:
            dt = (t1 - t0) 
            tokens = (loader.B * loader.T * grad_accum_steps * ddp_world_size) / dt
            print(f"step: {step}, loss: {loss_accum.item():.4f}, norm: {norm:.2f}, time: {1000 * dt:.2f} ms, tok/sec: {tokens:.2f}")
            t0 = time.time()

        if master_process and (step == max_iters):
            save_checkpoint(raw_model=raw_model, step=step)

if ddp:
    dist.destroy_process_group()
# getattr(model, "_orig_mod", model)
