
1.Why should we build our own LLMs ? 
we learn a lot while building our own custom LLMs that helps to understand from ground up of how does the llm works and while training a model it helps to understand its mechanic and limitation
custom made llms that are trained on custom data perform well than the generalized LLM like chatgpt 
also it equips with the knowledge of pretraining and finetuning the existing llm for our usecase


What is the process of creating an LLM? 

There are two stages of creating an LLM :
PreTraining:  

It refers to the training of the model in a large diverse dataset  to develop broad understanding of the language 
Pre-Trained model serves a foundation resources that can be further refined through  finetuning

FineTuning
After the model is pre-trained on  a large diverse dataset it is finetuned using smaller dataset for everyusecases that the model requires
for eg: the model can be finetuned to code by providing a dataset of coding

What is autoregressive model ? 
Autoregressive model are the model that uses the previous output as inputs to determine the future predictions 

what is the use of predicting the next token ? 
The use of predicing the next to key to gpt is predicts the next token according to the user queries
--------------------------------------------------------------------------------------------------------------------------------------------------------
What are the stages of building an LLM? 
The stages of building an LLM are 
Stage1: Building an LLM
1.Data preparation 
2.Attention Mechanism 
3.LLM Architecture 
 
Pretrain the model, the proceed to stage 2
Stage2 : Foundation model 
Training loop
Model Evalulation 
Save pre-trained weights 

Stage 3:
Finetuning the model according to a labeled data 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Working with text data 

Prepering text for large language model training 
Splitting text into word and subword tokens 
Byte pair encoding as a more advanced way of tokenizing text 
Sampling training examples with a sliding window 
Converting tokens into vectors that feed into a llm 



Word Embedding: 
what is problem ? why do we have to convert the data into embeddings ? what is embedding ? 
so the problem is neural network or transformer cannot understand plain text as so what we do is :
1.We turn the data into vectors using embedding model
2.The embedding model or pre-trained nn would be different for different types of data
3.Embedding is mapping the data into vectors in a countinous vector space. 

-------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
Tokenizing text 

how do you tokenize the text ? 
first , we have a set of corpus we divide the text into sub words and tokenize it for example "My name is Ripesh Ghimire" can be divided into ["my","name","is","Ripesh","Ghimire"]
keeping whitespaces or not is debatable

what is the usecase of having whitespaces and not having whitespaces ? 
having whitespaces is good for coding llm that does python because indentation is a big thing in python. So, it matters on the usecase


how do you convert token into token ids ?
we map every unique token to token id for eg the above token that we perfomed on my name is ripesh ghimire 
token ids can be
my'1' , name '2' etc 

why do we use special tokens ?  what are the type of special token that are used in tokenization ? 
we define a vocab_size, if the new word that we want to encode is not in the vocabulary, we need to encode it for example new word can be encoded as <|unk|> for unique words 
<|endoftext|> is used for defining the end of a data for example if you have two books we need to apply the end of text so the model understands that after this there is new text 
<|pad|> when training llms with batch size larger than one the bathch might contain pad to token to keep up to the lenght of the longest text in the batch
BOS :BEGGINING OF SEQUENCE
EOS END OF SEQUENCE 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
Byte Pair Encoding

Why Byte Pair Encoding ? 
breaks down word into sub word units 

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Data sampling with sliding window approach

Why sliding window ? 
sliding window approach in llm refers to making input and output pairs for the llm to train. Basically we will create input pair and output pair and train the model on that
[290] ----------> 4920
[290, 4920] ----------> 2241
[290, 4920, 2241] ----------> 287
[290, 4920, 2241, 287] ----------> 257

How do you do it ? 
we do it by making a batch_size and we create the batch size for example the batch size of 4 means the the four tokens are input for the llm and 5th token is the y value or target value for the llm 

why do you we need stride ? 
stirde helps the slide the input or output_tokens in the llm .If the stride is set to 1 we shift the input windo by 1 position whne creating the next,batch.
Note: If we set the stride equal to the input window size , we can prevent overlaps between the batches

what is the use of batch size ? 
to define the input and output for the llm 


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
creating embedding from token ids 

how do you creating embedding from tokenids ? 
we initialize a embedding vector. It is basically a lookup table for the token 

what is the use case of nn.Embedding ? 
nn.Embedding helps us to make embedding layer of vocab_size,d_model


Encoding word positions 
what is the limitation of self-attention in LLM?
The limitation of self attention is that there is no way to find the position of the data. To resolve the issue positional encoding was developed 

Positional Encoding are added to the token embedding vector to create the input embedding for an LLM 
The positional vectors embedding have the same demension as the original token embedding. 



'''“Figure 2.16 Embedding layers perform a look-up operation, retrieving the embedding vector corresponding to the token ID from the embedding layer's weight matrix. 
For instance, the embedding vector of the token ID 5 is the sixth row of the embedding layer weight matrix (it is the sixth instead of the fifth row because Python starts counting at 0). 
For illustration purposes, we assume that the token IDs were produced by the small vocabulary we used in section 2.3.'''



'''
Encoding word positions 
converted token ids into embedding , In principle this is a suitable input for an LLM. However a minor shortcoming of llms is that their self attention mechanism . doesn't have the notion of position or order for the order for the position
within a sequence 
'''


'''
“If we compare the embedding vector for token ID 3 to the previous embedding matrix, we see that it is identical to the 4th row (Python starts with a zero index, so it's the row corresponding to index 3). 
In other words, the embedding layer is essentially a look-up operation that retrieves rows from the embedding layer's weight matrix via a token ID.”


'''

LLMs require textual data to be converted into numerical vectors, known as embeddings since they can't process raw text. Embeddings transform discrete data (like words or images) into continuous vector spaces, making them compatible with neural network operations.
As the first step, raw text is broken into tokens, which can be words or characters. Then, the tokens are converted into integer representations, termed token IDs.
Special tokens, such as <|unk|> and <|endoftext|>, can be added to enhance the model's understanding and handle various contexts, such as unknown words or marking the boundary between unrelated texts.
The byte pair encoding (BPE) tokenizer used for LLMs like GPT-2 and GPT-3 can efficiently handle unknown words by breaking them down into subword units or individual characters.
We use a sliding window approach on tokenized data to generate input-target pairs for LLM training.
Embedding layers in PyTorch function as a lookup operation, retrieving vectors corresponding to token IDs. The resulting embedding vectors provide continuous representations of tokens, which is crucial for training deep learning models like LLMs.
While token embeddings provide consistent vector representations for each token, they lack a sense of the token's position in a sequence. To rectify[…]”


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Coding Attention Mechanism**  
**Exploring the reasons for using attention mechanisms in neural networks**  
**Introducing a basic self-attention framework and progressing to an enhanced self-attention mechanism**  
**Implementing a causal self-attention module that allows LLMs to generate one token at a time**  
**Masking randomly selected attention weights with dropout to reduce overfitting**  
**Stacking multiple causal attention modules into a multi-head attention**  

why attention mechanism ? 

what are the four different variants of self attention ? what do they do ? 
Simple attention 
self attention 
casual attention 
multihead attention 

why is traditional rnn not used ? 

why are self-attention introduced in ml? 

what is casual attention ? 

what does self in self attention mean ? 
