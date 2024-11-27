import numpy as np 
import math
import torch
import torch.nn as nn 

class InputEmbedding(nn.Module): 
    '''
    Input embedding: encodes the token or words passed into the transformer into a high dimensional vector like (512,768)
    d_model: represents the size of the embedding vector of each word 
    '''
    def __init__(self,d_model:int,vocab_size:int):
        """
        Initializes the InputEmbedding layer

        Parameters:
        d_model(int): represents the size of the embedding vector of each word
        vocab_size(int) : the size of the vocabulary (number of unique tokens)
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    def forward(self,x:torch.Tensor):

        '''
        Forward pass for the input embedding.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token indices.
                          Each value in the tensor is a token index corresponding to a word in the vocabulary.

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model), which is the input tokens 
                      mapped to high-dimensional embedding vectors, scaled by the square root of d_model.
        '''
        embedding_output = self.embedding(x)
        scaled_embedding_output = embedding_output * math.sqrt(self.d_model)
        return scaled_embedding_output
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float):

        '''
        Initialized the positional encoding layer in the transformer
            Parameters:
            d_model (int): The dimension of each word's embedding vector.
            seq_len (int): The length of the input sequence.
            dropout (float): The dropout rate applied after positional encoding to regularize the model.
        '''
        super.__init__()
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        #create a matrix of shape (seq_len,d_model)
        pe = torch.zeros(seq_len,d_model)
        #create a vecotr of shape seq_len
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        #for even position apply sin 
        pe[:,0::2] = torch.sin(position * div_term)
        #for odd position apply cos
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1,seq_len,d_model)
        self.register_buffer('pe',pe)
    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad(False)
        return self.dropout(x)
class LayerNormalization(nn.Module):
    """
    A PyTorch implementation of Layer Normalization, which normalizes the input along the last dimension.

    Attributes:
        eps (float): A small constant added to the standard deviation for numerical stability. Default is 1e-6.
        alpha (nn.Parameter): A learnable scaling parameter that adjusts the normalized input.
        bias (nn.Parameter): A learnable bias parameter added after normalization.
    
    Methods:
        forward(x):
            Performs layer normalization on the input tensor `x`.
    
    Example:
        >>> layer_norm = LayerNormalization(eps=1e-5)
        >>> x = torch.randn(2, 3)
        >>> output = layer_norm(x)
    
    """

    def __init__(self, eps: float = 1e-6) -> None:
        """
        Initializes the LayerNormalization layer.

        Args:
            eps (float): A small constant for numerical stability. Default is 1e-6.
        """
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(1))  # Scaling factor for normalized output
        self.bias = nn.Parameter(torch.zeros(1))  # Bias term added to normalized output

    def forward(self, x):
        """
        Applies layer normalization to the input tensor `x`.

        Args:
            x (torch.Tensor): Input tensor to normalize, with the last dimension normalized.

        Returns:
            torch.Tensor: The layer-normalized output tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
class FeedForwardNetwork(nn.Module):
    """
    A feed-forward neural network layer often used in transformer models. 
    This layer consists of two linear transformations with a ReLU activation function in between 
    and includes dropout for regularization. This architecture is common in many transformer models 
    as the position-wise feed-forward layer.

    Attributes:
        linear1 (nn.Linear): The first linear transformation (affine layer) which maps from 
            the input dimension (`d_model`) to a hidden layer of size `d_ff`.
        dropout (nn.Dropout): A dropout layer applied after the activation function to 
            prevent overfitting.
        linear2 (nn.Linear): The second linear transformation, which maps from the hidden 
            dimension (`d_ff`) back to the input dimension (`d_model`).

    Args:
        d_model (int): The input and output dimension of the model (e.g., the size of the embeddings).
        d_ff (int): The hidden dimension of the feed-forward layer.
        dropout (float): The dropout rate, a float between 0 and 1, indicating the probability 
            of an element to be zeroed during training.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # First linear layer, W1 and b1
        self.dropout = nn.Dropout(dropout)       # Dropout layer for regularization
        self.linear2 = nn.Linear(d_ff, d_model)  # Second linear layer, W2 and b2

    def forward(self, x):
        """
        Forward pass through the feed-forward network.

        Args:
            x (Tensor): The input tensor of shape (Batch, seq_len, d_model), 
                where `Batch` is the batch size, `seq_len` is the sequence length, 
                and `d_model` is the input dimension.

        Returns:
            Tensor: Output tensor of the same shape as the input (Batch, seq_len, d_model).
        """
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,heads:int,dropout:float,bias=False):
        self.d_model = d_model #embedding vector size