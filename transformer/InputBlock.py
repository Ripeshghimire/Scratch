import torch 
import math
import torch.nn as nn 
class InputEmbedding(nn.Module):
    """
    A custom PyTorch module for creating token embeddings for input sequences, scaled by the square root of the embedding dimension.

    Attributes:
    ----------
    vocab_size : int
        The size of the vocabulary. This determines the number of unique tokens that can be embedded.
    d_model : int
        The dimension of the embedding space. Each token is represented as a vector of this size.
    embedding : nn.Embedding
        The embedding layer used to map tokens to dense vectors of size `d_model`.

    Methods:
    -------
    __init__(self, vocab_size: int, d_model: int):
        Initializes the embedding layer with the specified vocabulary size and embedding dimension.

    forward(self, x: Tensor) -> Tensor:
        Computes the scaled embeddings for the input tokens.

        Parameters:
        ----------
        x : Tensor
            A tensor of token indices with shape `(batch_size, sequence_length)`.

        Returns:
        -------
        Tensor
            The scaled embedding tensor with shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(self, vocab_size, d_model):
        """
        Initializes the InputEmbedding module.

        Parameters:
        ----------
        vocab_size : int
            The size of the vocabulary. Determines the number of unique token embeddings.
        d_model : int
            The dimensionality of the embedding vectors.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass for the InputEmbedding module.

        This method takes a batch of token indices as input, computes their embeddings using
        the embedding layer, and scales them by the square root of the embedding dimension.

        Parameters:
        ----------
        x : Tensor
            A tensor of token indices with shape `(batch_size, sequence_length)`.

        Returns:
        -------
        Tensor
            The scaled embeddings with shape `(batch_size, sequence_length, d_model)`.
        """
        return self.embedding(x) * math.sqrt(self.d_model)
class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding as described in the "Attention Is All You Need" paper
    for Transformer models. This encoding helps the model incorporate the order of input sequences.

    Attributes:
        seq_len (int): The length of the input sequence.
        d_model (int): The dimensionality of the model's embeddings.
        pe (torch.Tensor): Precomputed positional encoding matrix of shape (1, seq_len, d_model).

    Methods:
        forward(x): Adds the positional encoding to the input tensor.

    Parameters:
        seq_len (int): The maximum length of the sequence for which positional encodings are generated.
        d_model (int): The dimensionality of the embeddings.

    Notes:
        Positional encoding uses sine and cosine functions of different frequencies
        to create a unique representation for each position in the sequence.
    """
    def __init__(self, seq_len, d_model):
        """
        Initializes the PositionalEncoding module.

        Args:
            seq_len (int): Length of the input sequence.
            d_model (int): Dimensionality of the embedding space.
        """
        super().__init__()
        self.seq_len = seq_len  
        self.d_model = d_model

        # Initialize the positional encoding matrix with zeros
        pe = torch.zeros(seq_len, d_model)
        
        # Create a tensor for positions (0 to seq_len - 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the division term for even indices
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices in the positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices in the positional encoding matrix
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension
        pe = pe.unsqueeze(0)
        
        # Register the positional encoding as a buffer (non-trainable parameter)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The input tensor with added positional encoding,
                        of shape (batch_size, seq_len, d_model).
        """
        return x + self.pe[:, :x.size(1), :]



    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "dmodel cannot be divided by h"
        self.d_key = d_model // h
        self.w_query= nn.Linear(d_model,d_model)
        self.w_key = nn.Linear(d_model,d_model)
        self.w_value = nn.Linear(d_model,d_model)
        self.w_output = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,query,key,value,mask):
        query = self.w_query(query) #(Batch, seq_len , d_model) -----> 
        key = self.w_key(key)
        value = self.w_value(value)
