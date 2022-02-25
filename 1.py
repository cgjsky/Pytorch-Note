import torch.nn as nn
import torch
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet,self).__init__()
        self.linear=nn.Linear()