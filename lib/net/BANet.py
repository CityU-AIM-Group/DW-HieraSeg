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

class BAA(nn.Module):
    # Cross Refinement Unit
    def __init__(self, cfg, E_channel, channel):
        super(BAA, self).__init__()
        self.semantic_encoder = nn.Sequential(
				nn.Conv2d(channel, channel, 1, 1, padding=0,bias=True),
				SynchronizedBatchNorm2d(channel, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
        )
        self.boundary_encoder = nn.Sequential(
				nn.Conv2d(E_channel, channel, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(channel, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
        )

        self.conv1 = nn.Sequential(
				nn.Conv2d(channel, channel, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(channel, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
				nn.Conv2d(channel, channel, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(channel, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
        )

        self.cls_semantic_conv = nn.Conv2d(channel, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        self.cls_bounary_conv = nn.Conv2d(channel, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample2_ = nn.UpsamplingBilinear2d(scale_factor=2)
        self.BAA_out = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(channel, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
        )

    def forward(self, xe, xd, is_upsample = True):
        S1 = self.semantic_encoder(xd)
        B1 = self.boundary_encoder(xe)

        SB1 = self.conv1(S1.mul(B1))
        S2 = S1 + SB1
        B2 = B1 + SB1

        SB2 = self.conv2(S2.mul(B2))
        S3 = S2 + SB2
        B3 = B2 + SB2

        if is_upsample == True:
            S3 = self.upsample2(S3)
            B3 = self.upsample2_(B3)
        semantic_cls = self.cls_semantic_conv(S3)
        boundary_cls = self.cls_bounary_conv(B3)
        output = self.BAA_out(S3)

        return output, semantic_cls, boundary_cls

class BANet(nn.Module):
	def __init__(self, cfg):
		super(BANet, self).__init__()
		self.backbone = None		
		self.backbone_layers = None
		input_channel = 2048		
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=cfg.MODEL_ASPP_OUTDIM, 
				rate=16//cfg.MODEL_OUTPUT_STRIDE,
				bn_mom = cfg.TRAIN_BN_MOM)
		self.dropout1 = nn.Dropout(0.5)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)

		self.D3_conv = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
        )

		self.BAA3to2 = BAA(cfg, input_channel, cfg.MODEL_ASPP_OUTDIM)
		self.BAA2to1 = BAA(cfg, 1024, cfg.MODEL_ASPP_OUTDIM)
		self.BAA1to0 = BAA(cfg, 512, cfg.MODEL_ASPP_OUTDIM)

		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
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
		E1 = layers[1]
		E2 = layers[2]
		E3 = layers[3] ## layers[3] = layers[-1]
		feature_aspp = self.aspp(layers[-1])
		D3 = self.dropout1(feature_aspp)

	#	D3 = self.D3_conv(feature_aspp)
		D2, sem_cls2, bou_cls2  = self.BAA3to2(E3, D3, is_upsample = False)
		D1, sem_cls1, bou_cls1  = self.BAA2to1(E2, D2)
		D0, sem_cls0, bou_cls0  = self.BAA1to0(E1, D1)

		result = self.cls_conv(D0)
		result = self.upsample4(result)
		BAA_loss_list = {'semantic_cls2_16':sem_cls2, 'boundary_cls2_16':bou_cls2, 
						'semantic_cls1_32':sem_cls1, 'boundary_cls1_32':bou_cls1, 
						'semantic_cls0_64':sem_cls0, 'boundary_cls0_64':bou_cls0}

		return result, BAA_loss_list

