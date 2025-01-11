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
import torch
import torch.nn as nn
import math

class LoraConfig:
    def __init__(self, features_in, features_out, rank=1, alpha=1.0):
        self.rank = rank
        self.features_in = features_in
        self.features_out = features_out
        self.alpha = alpha  # Scaling factor for stability

class LoraLayer(nn.Module):
    def __init__(self, config: LoraConfig):
        super().__init__()
        
        self.config = config
        scaling = self.config.alpha / self.config.rank
        
        # Initialize A and B matrices
        # A: down-projection matrix (features_in → rank)
        # B: up-projection matrix (rank → features_out)
        self.lora_A = nn.Parameter(
            torch.zeros(self.config.features_in, self.config.rank)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.config.rank, self.config.features_out)
        )
        self.scaling = scaling
        
        # Initialize weights using gaussian distribution
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using kaiming uniform initialization"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # Compute the LoRA transformation: (x @ A) @ B * scaling
        return (x @ self.lora_A) @ self.lora_B * self.scaling

class LoraLinear(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1.0, bias=True):
        super().__init__()
        
        # Regular linear layer
        self.linear = nn.Linear(features_in, features_out, bias=bias)
        
        # LoRA components
        self.lora = LoraLayer(
            LoraConfig(
                features_in=features_in,
                features_out=features_out,
                rank=rank,
                alpha=alpha
            )
        )
        
    def forward(self, x):
        # Combine the regular linear transformation with the LoRA update
        return self.linear(x) + self.lora(x)

# Example usage
def example_usage():
    # Create a LoRA linear layer
    features_in, features_out = 768, 512  # Example dimensions
    rank = 8  # LoRA rank
    
    # Initialize layer
    lora_layer = LoraLinear(
        features_in=features_in,
        features_out=features_out,
        rank=rank,
        alpha=16
    )
    
    # Create sample input
    batch_size = 32
    x = torch.randn(batch_size, features_in)
    
    # Forward pass
    output = lora_layer(x)
    
    # Output shape should be [batch_size, features_out]
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of trainable parameters in LoRA: "
          f"{sum(p.numel() for p in lora_layer.lora.parameters())}")
    print(f"Number of trainable parameters in base linear: "
          f"{sum(p.numel() for p in lora_layer.linear.parameters())}")

if __name__ == "__main__":
    example_usage()