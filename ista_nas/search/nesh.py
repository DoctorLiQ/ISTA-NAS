import torch
import numpy as np
import torch.nn as nn

__all__ = ["Nesh"]

class Nesh:
    def step(self,Acc,normal,reduce):
        normal = torch.Tensor([[0,5,0.4,0.1],[0.6,0.2,0.2],[0.3,0.5,0.2],[0.1,0.9,0.0]])
        reduce = torch.Tensor([[0,5,0.4,0.1],[0.6,0.2,0.2],[0.3,0.5,0.2],[0.1,0.9,0.0]])