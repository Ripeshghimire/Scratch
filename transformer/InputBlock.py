import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x:torch.Tensor):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self,seq_len:int,d_model:int,dropout:float):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = nn.Embedding(seq_len,d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)
        #Register the positional encoding as a buffer
        self.register_buffer('pe', pe)
    def forward(self,x:torch.Tensor):
        x = x + self.pe[:, :x.size(1),:].requires_grad(False)
        return self.dropout(x)

