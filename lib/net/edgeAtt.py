# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from net.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from net.backbone import build_backbone
from net.ASPP import ASPP

class EdgeEnhancing(nn.Module):
    # Cross Refinement Unit
    def __init__(self, cfg, channel):
        super(EdgeEnhancing, self).__init__()
        self.conv1_1 = nn.Conv2d(channel, channel, 1, 1, padding=0)
        self.conv1_2 = nn.Conv2d(channel, channel, 1, 1, padding=0)
        self.edge_encoder = nn.Sequential(
				nn.Conv2d(channel, channel, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(channel, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
        )
        self.conv3_1 = nn.Sequential(
				nn.Conv2d(channel, channel, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(channel, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
        )
        self.conv3_2 = nn.Sequential(
				nn.Conv2d(channel, channel, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(channel, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
        )
        self.conv3_3 = nn.Sequential(
				nn.Conv2d(channel, channel, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(channel, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
        )
        self.cls_sem_conv = nn.Conv2d(channel, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        self.cls_edge_conv = nn.Conv2d(channel, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)

    def forward(self, x):
        S1 = self.conv1_1(x)
        E0 = self.conv1_2(x)
        E1 = self.edge_encoder(E0)

        SE1 = self.conv3_1(S1.mul(E1))
        S2 = S1 + SE1
        E2 = E1 + SE1

        SE2 = self.conv3_2(S2.mul(E2))
        S3 = S2 + SE2
        E3 = E2 + SE2

#        SE3 = self.conv3_3(S3.mul(E3))
#        output = S3 + SE3

        sem_cls = self.cls_sem_conv(S3)
        edge_cls = self.cls_edge_conv(E3)

        return S3, sem_cls, edge_cls

class edgeAtt(nn.Module):
	def __init__(self, cfg):
		super(edgeAtt, self).__init__()
		self.backbone = None		
		self.backbone_layers = None
		input_channel = 2048		
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=cfg.MODEL_ASPP_OUTDIM, 
				rate=16//cfg.MODEL_OUTPUT_STRIDE,
				bn_mom = cfg.TRAIN_BN_MOM)
		self.dropout1 = nn.Dropout(0.5)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)

		indim = 256
		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL//2,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),		
		)		

		self.edge_attention = EdgeEnhancing(cfg, cfg.MODEL_SHORTCUT_DIM+cfg.MODEL_ASPP_OUTDIM)
		self.cat_conv = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, SynchronizedBatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
		self.backbone_layers = self.backbone.get_layers()

	def forward(self, x):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)
		feature_aspp = self.upsample_sub(feature_aspp)

		feature_shallow = self.shortcut_conv(layers[0])
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)

		output, sem_cls, edge_cls = self.edge_attention(feature_cat)

		feature = self.cat_conv(output) 
		result = self.cls_conv(feature)
		result = self.upsample4(result)
		return result, sem_cls, edge_cls

