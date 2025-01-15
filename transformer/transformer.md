Transformer is a architecture that was proposed by Google in 2017 in the "Attention is all you Need" paper. It is the backbone of modern
 LLM like GPT,BERT,Claude.

Transformer consists of two layers: 
1. Encoder
Encoder in the transformer is responsible for processing the input text into embedding through different layers in the transformers(which we will talk about in a while). 
It captures the contextual information of the input sentence. For example in a machine translation task we have a word what is your name in english the primary work of the encoder is to convert this set of text to mathematical numbers and pass it to the decoder.

2. Decoder
    So now we have the mathematical numbers the numbers are passed through the decoder to generate the desired output we wanted. According the the previous example we are converting the given text to nepali the output of that text would be timro naam k ho ?
these are the primary operation that happens in the transformer 

Now, Let's define each block and what we do it with it: 
There are 
1.Input Embedding 
2.Masked/Multi-Head Attentino 
3.Positional Encoding 
4.Feed Forward Layer 
5.Add and Norm Layer i.e Layer Normalization 
6.Residual Connection

All of these layer are used in both the encoder and decoder 

Let's talk about the Input Embedding layer: 

Input Embedding : 
First we convert the data into the text into the tokens after that, what we do is we define a vocab_size for the model to make a lookup table for the 
input data and make a embedding layer that is the embedding layer would be of (vocab_size,d_model)
Input Embedding is something that we use to convert the nn.

In PyTorch is used to define as: 
class InputEmbedding(nn.Module):
    def __init__(self,d_model,vocab_size):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.Embedding = nn.Embedding(vocab_size,d_model)
    def forward(x):
        return self.embedding(x) * math.sqrt(d_model)
    
According to the paper the embedding layer is divided by square of d_model 

PositionalEncoding
This layer in transformer is used to 
        

