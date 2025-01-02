'''
Fine tuning mean training a pre-trained network on new data to improve its perfomance
on a specific task. For example, we may fine-tune a LLM that was trained on mamy programming 
language and fine tune it for a new dialect of SQL
Problems with finetuning: 
1.train the full network, which it computainoally expensive for the average user when dealing with large models
2.storage requirements for the checkpoints are expensive, as we need to save the entire model on disk 
for each checkpoint. 
3.if we have multiple fine-tuned models, we need tor reaload all the weights of the model every time,we want 
to switch between them.
'''
'''
Introduction LORA:

1.start with the input ---> we have our pretrained model that has its weights(we want to fine-tune it: we want to freeze the weights(using pyTorch))
we never run backpropagation on them. 

2.Then we create two other matrices one for each other layers one for each of the metrics that we want to train so basically in lora we don't have to 
Understanding LORA: utilized low-rank representations to adapt large models,enabling efficient fine-tuning without altering the original parameter

Importance of LoRA:
Parameter Efficiency: The reduction in trainable parameters leads to faster backpropagation and low storage needs,
enhancing overall model efficiency 

Model Switching: LoRA allows seamless transitions between fine-tuned models, which is advantegous us for applications requiring varied outputs.

Preserving Original Weights: Keeping the original model weights intact ensures that pre-trained capabilities are mantained while enabling specific adaptations 

📉 LoRA Benefits: LoRA reduces parameters, storage needs, and speeds up backpropagation.

'''
import torch
import torch.nn as nn 
class LoraConfig(nn.Module):
    def __init__(self,features_in,features_out,rank=1):
        self.rank = rank 
        self.features_in = features_in 
        self.features_out =  features_out
    