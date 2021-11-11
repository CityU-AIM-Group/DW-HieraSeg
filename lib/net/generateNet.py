# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from net.deeplabv3plus import deeplabv3plus
from net.deeplabv3plus_meta import deeplabv3plus_meta, LearnSPL_meta
from net.learnSPL1 import LearnSPL
from net.fcn import FCN8s
from net.unetplusplus import NestedUNet
from net.SegNet import SegNet

def generate_net(cfg, is_learning = None):
	if is_learning:
		if cfg.MODEL_NAME == 'deeplabv3plus' or cfg.MODEL_NAME == 'deeplabv3+':
			return LearnSPL(cfg)
		if cfg.MODEL_NAME == 'deeplabv3plus_meta' or cfg.MODEL_NAME == 'deeplabv3+_meta':
			return LearnSPL_meta(cfg)
	else: 
		if cfg.MODEL_NAME == 'deeplabv3plus' or cfg.MODEL_NAME == 'deeplabv3+':
			return deeplabv3plus(cfg)
		if cfg.MODEL_NAME == 'deeplabv3plus_meta' or cfg.MODEL_NAME == 'deeplabv3+_meta':
			return deeplabv3plus_meta(cfg)
	if cfg.MODEL_NAME == 'fcn' or cfg.MODEL_NAME == 'FCN':
		return FCN8s(cfg)
	if cfg.MODEL_NAME == 'UNetplusplus' or cfg.MODEL_NAME == 'UNet++':
		return NestedUNet(cfg)
	if cfg.MODEL_NAME == 'segnet' or cfg.MODEL_NAME == 'SegNet':
		return SegNet(cfg)
	else:
		raise ValueError('generateNet.py: network %s is not support yet'%cfg.MODEL_NAME)
