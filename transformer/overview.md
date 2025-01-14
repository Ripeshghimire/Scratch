Transformer is a architecture that was proposed by Google in 2017. It is the backbone of modern LLM like GPT,BERT,Claude.

Transformer consists of two layers: 
1. Encoder
    Encoder in the transformer is responsible for processing the input text into embedding through different layers in the transformers(which we will talk about in a while). It captures the contextual information of the input sentence. For example in a machine translation task we have a word what is your name in english the primary work of the encoder is to convert this set of text to mathematical numbers and pass it to the decoder.

2. Decoder
    So now we have the mathematical numbers the numbers are passed through the decoder to generate the desired output we wanted. According the the previous example we are converting the given text to nepali the output of that text would be timro naam k ho ? 

these are the primary operation that happens in the transformer 

Now, Let's define each block and what we do it with it: 
There are 
1.Input Embedding 
2.Masked/Multi-Head Attentino 
3.Positional Encoding 
4.