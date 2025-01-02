import torch
import torch.nn as nn 
class GELU(nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self,x):
        return 0.5 * x * torch.tanh(torch.sqrt(2 / torch.pi) * (x + torch.pow(0.44715, 3)))


    