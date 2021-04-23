import torch
import numpy as np
import torch.nn as nn

__all__ = ["OuterTrainer"]

class OuterTrainer:

    def __init__(self, model, args):
        #self.network_momentum = args.momentum
        #self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def step(self,  input_valid, target_valid):
        self.optimizer.zero_grad()
        self._backward_step(input_valid, target_valid)
        self.optimizer.step()
    

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward(retain_graph = True)
