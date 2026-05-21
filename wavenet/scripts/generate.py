import torch
from wavenet import WaveNet 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = WaveNet()
model.to(device=device)
model.load_state_dict(torch.load("b16e32h64.pt", weights_only=True))

for _ in range(10):
    print("".join(model.generate()))