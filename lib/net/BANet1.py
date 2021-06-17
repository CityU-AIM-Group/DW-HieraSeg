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

def make_one_hot(labels):
	target = torch.eye(cfg.MODEL_NUM_CLASSES)[labels]
	gt_1_hot = target.permute(0, 3, 1, 2).float().cuda()
	return gt_1_hot

class BAA(nn.Module):
    # Cross Refinement Unit
    def __init__(self, cfg, in_channel, channel, pool_win = 3):
        super(BAA, self).__init__()
        self.semantic_encoder = nn.Conv2d(in_channel, channel, 1, 1, padding=0,bias=True)
        self.boundary_encoder1 = nn.Conv2d(in_channel, channel, 1, 1, padding=0,bias=True)
        self.boundary_encoder2 = nn.Conv2d(in_channel, channel, 1, 1, padding=0,bias=True)
        self.avgpool = nn.AvgPool2d(pool_win, stride=1, padding=pool_win//2)

        self.conv1 = nn.Sequential(
				nn.Conv2d(channel, channel, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(channel, momentum=cfg.TRAIN_BN_MOM),
#				nn.Dropout(0.5),
        )
        self.conv1_1 = nn.Sequential(
				nn.Conv2d(channel, channel, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(channel, momentum=cfg.TRAIN_BN_MOM),
#				nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
				nn.Conv2d(channel, channel, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(channel, momentum=cfg.TRAIN_BN_MOM),
#				nn.Dropout(0.5),
        )

        self.cls_semantic_conv = nn.Conv2d(channel, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        self.cls_bounary_conv = nn.Conv2d(channel, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)

    def forward(self, xe, is_upsample = True):
        S1 = self.semantic_encoder(xe)
    #    B1 = self.boundary_encoder(xe)

        B = self.boundary_encoder1(xe)
        B_ = self.boundary_encoder2(xe)
        B1 = (1-torch.sigmoid(B)).mul(B_)

    #    B = self.boundary_encoder1(xe)
    #    B1 = B - self.avgpool(B)

        S2 = S1 + S1 * torch.sigmoid(self.conv1(S1 + B1))
#        S2 = S1 + S1 * torch.sigmoid(self.conv1(torch.cat([S1,B1],1)))
        B2 = B1 + B1 * torch.sigmoid(self.conv1_1(S1 * B1))

#        S3 = S2 + S2 * torch.sigmoid(self.conv2(torch.cat([S2,B2],1)))
        S3 = S2 + S2 * torch.sigmoid(self.conv2(S2 + B2))
#        B3 = B2 + B2 * torch.sigmoid(self.conv2_1(S2 * B2))

        semantic_cls = self.cls_semantic_conv(S3)
        boundary_cls = self.cls_bounary_conv(B2)
        return S3, semantic_cls, boundary_cls

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
		self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)

		self.BAA2 = BAA(cfg, 1024, cfg.MODEL_SHORTCUT_DIM, pool_win = 7)
		self.BAA1 = BAA(cfg, 512, cfg.MODEL_SHORTCUT_DIM, pool_win = 5)
		self.BAA0 = BAA(cfg, 256, cfg.MODEL_SHORTCUT_DIM, pool_win = 3)

		self.imgresize = torch.nn.UpsamplingNearest2d(size=(int(cfg.DATA_RESCALE//4),int(cfg.DATA_RESCALE//4)))
		self.fore_centers = nn.Parameter(torch.randn(cfg.MODEL_SHORTCUT_DIM*3 + cfg.MODEL_ASPP_OUTDIM).cuda(), requires_grad=False)
		self.back_centers = nn.Parameter(torch.randn(cfg.MODEL_SHORTCUT_DIM*3 + cfg.MODEL_ASPP_OUTDIM).cuda(), requires_grad=False)

		self.cat_conv = nn.Sequential(
				nn.Conv2d(cfg.MODEL_SHORTCUT_DIM*3 + cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
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

	def forward(self, x, y = None, lambda_=0.1, phase = 'test'):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		E0 = layers[0]
		E1 = layers[1]
		E2 = layers[2]
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)

		ED2, sem_cls2, bou_cls2  = self.BAA2(E2)
		ED1, sem_cls1, bou_cls1  = self.BAA1(E1)
		ED0, sem_cls0, bou_cls0  = self.BAA0(E0)

		D2 = torch.cat([feature_aspp, ED2],1)
		D1 = torch.cat([self.upsample2(D2), ED1],1)
		D0 = torch.cat([self.upsample2(D1), ED0],1)

		if phase == 'train':
			y = self.imgresize(torch.unsqueeze(y.float(), 1))
			fore_feature = torch.sum(D0*y, dim=(0,2,3))/torch.sum(y)
			back_feature = torch.sum(D0*(1.-y), dim=(0,2,3))/torch.sum(1.-y)
			self.fore_centers.data = (1-lambda_) * self.fore_centers.data + lambda_ * fore_feature
			self.back_centers.data = (1-lambda_) * self.fore_centers.data + lambda_ * back_feature
		fore_back = torch.sigmoid(self.fore_centers - self.back_centers)
		fore_back = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(fore_back, 0), 2), 2)

		D0 = D0.mul(fore_back)

		result = self.cat_conv(D0) 
		result = self.cls_conv(result)
		result = self.upsample4(result)
		BAA_loss_list = {'semantic_cls2_16':sem_cls2, 'boundary_cls2_16':bou_cls2, 
						'semantic_cls1_32':sem_cls1, 'boundary_cls1_32':bou_cls1, 
						'semantic_cls0_64':sem_cls0, 'boundary_cls0_64':bou_cls0}

		return result, BAA_loss_list

