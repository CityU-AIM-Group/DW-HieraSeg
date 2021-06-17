# ----------------------------------------
# Written by Xiaoqing Guo
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Mentor(nn.Module):
    def __init__(self, cfg):
        super(Mentor, self).__init__()

        #self.c1 = c1
        #self.c2 = c2 
        self.label_emb = nn.Embedding(cfg.MODEL_NUM_CLASSES, 2)
        self.percent_emb = nn.Embedding(100, 5)

        self.lstm = nn.LSTM(2, 10, 1, bidirectional=True)
        # self.h0 = torch.rand(2,)
        self.fc1 = nn.Linear(27, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, data):
        label, pt, l = data
        x_label = self.label_emb(label)
        x_percent = self.percent_emb(pt)
        # print(x_label.size())
        h0 = torch.rand(2, label.size(0), 10)
        c0 = torch.rand(2, label.size(0), 10)

        output, (hn,cn) = self.lstm(l, (h0,c0))
        output = output.sum(0).squeeze()
        x = torch.cat((x_label, x_percent, output), dim=1)
        z = F.tanh(self.fc1(x))
        z = F.sigmoid(self.fc2(z))
        return z