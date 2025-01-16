
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
---------------------------------------------------------------------------------------------------------------------------------------------------------------
What are the stages of building an LLM? 
The stages of building an LLM are 
Stage1: Building an LLM
1.Data preparation 
2.Attention Mechanism 
3.LLM Architecture 

Now we pretrain the model,
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

what is the usecase of having whitespaces and not having whitespaces ? 

how do you convert token into token ids ?
