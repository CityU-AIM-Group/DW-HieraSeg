# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from net.deeplabv3plus import deeplabv3plus
from net.deeplabv3plus_meta import deeplabv3plus_meta, LearnSPL_meta
from net.edgeAtt import edgeAtt
from net.BANet2 import BANet
#from net.deeplabv3plus_meta import LearnSPL
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
	if cfg.MODEL_NAME == 'edgeAtt':
		return edgeAtt(cfg)
	if cfg.MODEL_NAME == 'BANet':
		return BANet(cfg)
	if cfg.MODEL_NAME == 'fcn' or cfg.MODEL_NAME == 'FCN':
		return FCN8s(cfg)
	if cfg.MODEL_NAME == 'UNetplusplus' or cfg.MODEL_NAME == 'UNet++':
		return NestedUNet(cfg)
	if cfg.MODEL_NAME == 'segnet' or cfg.MODEL_NAME == 'SegNet':
		return SegNet(cfg)
	# if cfg.MODEL_NAME == 'supernet' or cfg.MODEL_NAME == 'SuperNet':
	# 	return SuperNet(cfg)
	# if cfg.MODEL_NAME == 'eanet' or cfg.MODEL_NAME == 'EANet':
	# 	return EANet(cfg)
	# if cfg.MODEL_NAME == 'danet' or cfg.MODEL_NAME == 'DANet':
	# 	return DANet(cfg)
	# if cfg.MODEL_NAME == 'deeplabv3plushd' or cfg.MODEL_NAME == 'deeplabv3+hd':
	# 	return deeplabv3plushd(cfg)
	# if cfg.MODEL_NAME == 'danethd' or cfg.MODEL_NAME == 'DANethd':
	# 	return DANethd(cfg)
	else:
		raise ValueError('generateNet.py: network %s is not support yet'%cfg.MODEL_NAME)
