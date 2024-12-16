import torch 
import math
import torch.nn as nn 
class InputEmbedding:
    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding:
    def __init__(self,seq_len,d_model):
        self.seq_len = seq_len  
        self.d_model = d_model
        pe = torch.zeros(seq_len,d_model)
        postion = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))