import torch

torch.cuda.init()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
a = torch.Tensor(1).to(device)
b = torch.Tensor(1).to(device)
print(a + b)


