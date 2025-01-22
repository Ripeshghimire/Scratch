import torch
import torch.nn as nn 
from torch.utils.data import Dataset,DataLoader
import tiktoken

class GPTDataset(Dataset):
    def __init__(self,text,tokenizer,max_length,stride):
        self.tokenizer = tokenizer
        self.input_ids =[]
        self.output_ids =[]
        token_ids = tokenizer.encode(text)
        #Ensure sequence are of same length 
        for i in range(0,len(token_ids) - max_length,stride):
            input_chunk = token_ids[i:i+max_length]
            output_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(input_chunk)
            self.output_ids.append(output_chunk)
    def __len__(self):
        '''
        __len__: Tells the DataLoader how many samples are in the dataset.
        '''
        return len(self.input_ids)
    def __getitem__(self, index):
        '''
        The __getitem__ function retrieves a specific sample from the dataset based on an index. This is used by the DataLoader to:

        Fetch individual samples.

        Construct batches (by calling __getitem__ multiple times).
        7. Key Points
        __len__: Tells the DataLoader how many samples are in the dataset.

        __getitem__: Retrieves a specific sample (input-output pair) by index.

        These functions are required for the DataLoader to work with your dataset.
        '''
        return self.input_ids[index],self.output_ids[index]
    

def create_loader_v1(text,batch_size=4,max_length = 256,stride = 128,shuffle=True,drop_last = True):
    tokenizer = tiktoken.get_encoding('gpt2') 
    dataset = GPTDataset(text,tokenizer,max_length,stride)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last)
    return dataloader
   