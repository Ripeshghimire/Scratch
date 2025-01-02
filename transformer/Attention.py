import torch 
import torch.nn as nn 

class MultiHeadAttention(nn.Module): 
    def __init__(self,d_model:int,d_ff:int,dropout:float,h:int)->None:
        self.h = h #Number of heads 
        self.d_model = d_model #Embedding vector size
        self.dropout = nn.Dropout()
        assert d_model * h ==0 ,"dmodel is not divisble by h "
        self.d_k = d_model //h #dimension of vector seen by each head 
        self.key = nn.Linear(d_model,d_model,bias=False)
        self.query = nn.Linear(d_model,d_model,bias=False)
        self.value = nn.Linear(d_model,d_model,bias=False)
        self.output = nn.Linear(d_model,d_model,bias=False)
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]
        
                  

