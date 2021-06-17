# ----------------------------------------
# Written by Xiaoqing Guo
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from net.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init

class LearnSPL(nn.Module):
    def __init__(self, cfg):
        super(LearnSPL, self).__init__()

        self.cfg = cfg
        self.label_emb = nn.Embedding(cfg.MODEL_NUM_CLASSES, 2)
        self.JA_percent_emb = nn.Embedding(101, 4)
        self.CE_percent_fore_emb = nn.Embedding(101, 4)
        self.CE_percent_back_emb = nn.Embedding(101, 4)

        self.lstm = nn.LSTM(2, 10, 1, bidirectional=True)

        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 1)

        self.weight_conv = nn.Sequential(
				nn.Conv2d(259, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
            #    nn.Tanh(),
		)
        self.weight_cls_conv = nn.Sequential(
                nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, 1, 1, 1, padding=0),
            #    nn.Sigmoid(),
        )
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, JA_loss, JA_percent, CE_loss, CE_percent, feature_maps, labels):        
        ### image-level weight
    #    print(JA_loss.shape)
    #    x_percent = self.JA_percent_emb((JA_percent*100*torch.ones(JA_loss.size(0))).int().long().cuda())
        x_percent = (JA_percent*torch.ones(JA_loss.size(0),1)).float().cuda()
    #    x_lambda = (lambda_JA*torch.ones(JA_loss.size(0),1)).float().cuda()
        JA_loss = torch.unsqueeze(JA_loss, 1).cuda()
        x = torch.cat((JA_loss, x_percent), dim=1)
    #    print(x_percent.shape)
    #    h0 = torch.rand(2, JA_loss.size(0), 10)
    #    c0 = torch.rand(2, JA_loss.size(0), 10)
    #    lstm_output, (hn,cn) = self.lstm(JA_loss, (h0,c0))
    #    lstm_output = lstm_output.sum(0).squeeze()  
    #    print(lstm_output.shape) 
    #    print(x)
        img = torch.tanh(self.fc1(x))
        img_weight = torch.sigmoid(self.fc2(img))

        ### pixel-level weight
    #    print(feature_maps.shape, labels.shape, CE_loss.shape)
        imgresize = torch.nn.UpsamplingBilinear2d(size=(int(self.cfg.DATA_RESCALE/4),int(self.cfg.DATA_RESCALE/4)))
        labels = imgresize(torch.unsqueeze(labels, 1).float())
        CE_loss = imgresize(torch.unsqueeze(CE_loss, 1))
    #    x_percent = (CE_percent*torch.ones(CE_loss.shape)).float().cuda()
        x_percent = imgresize(torch.unsqueeze(CE_percent, 1))
        feature_cat = torch.cat([feature_maps,labels,CE_loss,x_percent],1)
    #    print(feature_cat.shape)
        result = torch.tanh(self.weight_conv(feature_cat))
        result = torch.sigmoid(self.weight_cls_conv(result))
        pix_weight = self.upsample4(result)

        return img_weight, pix_weight