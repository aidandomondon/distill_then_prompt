from transformers.models.gpt2 import GPT2LMHeadModel
import torch
from os import listdir

print(torch.accelerator.device_count())
print(torch.accelerator.current_accelerator().type)

class A:
    @classmethod
    def from_pretrained(cls):
        print('Instantiating from pretrained.')

class B(A):
    def __init__():
        super().__init__()
        print('Child initializer called.')

b = B.from_pretrained()