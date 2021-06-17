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

class deeplabv3plus(nn.Module):
	def __init__(self, cfg):
		super(deeplabv3plus, self).__init__()
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
	#	self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		self.cls_conv_0 = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.normal_dim0 + cfg.polyp_dim0, 1, 1, padding=0, bias=False)
		self.cls_conv_1 = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.normal_dim1 + cfg.polyp_dim1, 1, 1, padding=0, bias=False)
		self.cls_conv_2 = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.normal_dim2 + cfg.polyp_dim2, 1, 1, padding=0, bias=False)

	#	self.imgresize = torch.nn.UpsamplingNearest2d(size=(int(cfg.DATA_RESCALE//4),int(cfg.DATA_RESCALE//4)))
	#	self.fore_centers = nn.Parameter(torch.randn(cfg.MODEL_SHORTCUT_DIM+cfg.MODEL_ASPP_OUTDIM).cuda(), requires_grad=False)
	#	self.back_centers = nn.Parameter(torch.randn(cfg.MODEL_SHORTCUT_DIM+cfg.MODEL_ASPP_OUTDIM).cuda(), requires_grad=False)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, SynchronizedBatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
		self.backbone_layers = self.backbone.get_layers()

	def forward(self, x, y = None, lambda_=0.1, FC_pyramid = False, phase = 'test'):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)
		feature_aspp = self.upsample_sub(feature_aspp)

		feature_shallow = self.shortcut_conv(layers[0])
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)

		'''if phase == 'train':
			y = self.imgresize(torch.unsqueeze(y.float(), 1))
			fore_feature = torch.sum(feature_cat*y, dim=(0,2,3))/torch.sum(y)
			back_feature = torch.sum(feature_cat*(1.-y), dim=(0,2,3))/torch.sum(1.-y)
			self.fore_centers.data = (1-lambda_) * self.fore_centers.data + lambda_ * fore_feature
			self.back_centers.data = (1-lambda_) * self.fore_centers.data + lambda_ * back_feature
		fore_back = torch.sigmoid(self.fore_centers - self.back_centers)
		fore_back = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(fore_back, 0), 2), 2)
		feature_cat = feature_cat.mul(fore_back)'''

		feature = self.cat_conv(feature_cat) 

		if FC_pyramid:
			result_0 = self.cls_conv_0(feature)
			result_0 = self.upsample4(result_0)
			result_1 = self.cls_conv_1(feature)
			result_1 = self.upsample4(result_1)
			result_2 = self.cls_conv_2(feature)
			result_2 = self.upsample4(result_2)
			return feature, result_0, result_1, result_2
		else:
			result = self.cls_conv(feature)
			result = self.upsample4(result)
			return feature, result

