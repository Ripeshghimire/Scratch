import torch
import torch.nn as nn
'''
Simple self attention: technique that introduces the broad idea of self attention
Self attention: trainable weights that from the basis of the mechanism used in LLMs
Causal attention : type of self attention used in LLMs that allows the model to only consider previous and current inputs
MultiHead attention: extension of self attention that enables the model to simultaneously attend information from different
representative subprocess
'''



class MultiHeadAttention(nn.Module):
    def __init__(self,d_in:int,d_out:int    ,h:int,dropout:float):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.h = h
        self.dropout = nn.Dropout(dropout)
        w_query = nn.Linear(d_in,d_out)
        w_key = nn.Linear(d_in,d_out)
        w_value = nn.Linear(d_in,d_out)
        w_output = nn.Linear(d_in,d_out)
