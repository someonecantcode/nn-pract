import torch
import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')
class DataLoader():
    def __init__(self, B, T, gpu_rank, total_gpus):
        self.B = B
        self.T = T
        self.gpu_rank = gpu_rank
        self.total_gpus = total_gpus

        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9*len(data)) # first 90% will be train, rest val
        self.train_data = data[:n]
        self.val_data = data[n:]

        self.current_pos = self.B * self.T * gpu_rank
        if gpu_rank == 0:
            print(f"tokens: {len(self.train_data)}")
            print(f"1 epoch is {len(self.train_data) // (self.B * self.T * total_gpus)} iters")

    def get_batch(self, split): # outdated(still works) random batch from gpt1
        batch_size, block_size = self.B, self.T
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(low=0, high=len(data)-block_size, size=(batch_size, ))

        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y
    
    def next_batch(self, split):
        B, T = self.B, self.T
        data = self.train_data if split == 'train' else self.val_data
        buffer = data[self.current_pos : self.current_pos + B*T + 1]
        
        x = buffer[:-1].view(B, T)
        y = buffer[1:].view(B, T)

        self.current_pos += B * T * self.total_gpus
        if self.current_pos >= (len(data) - (1+B*T)):
            self.current_pos = self.B * self.B * self.gpu_rank
        return x, y
