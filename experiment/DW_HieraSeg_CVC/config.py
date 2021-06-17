# ----------------------------------------
# Written by Xiaoqing Guo
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

class Configuration():
	def __init__(self):
		self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__"),'..','..'))
		self.CBloss_weight = 1.0 # <= 1
		self.normal_dim0 = 3
		self.normal_dim1 = 2
		self.normal_dim2 = 1
		self.polyp_dim0 = 9
		self.polyp_dim1 = 4
		self.polyp_dim2 = 1
		self.percent_fore_start = 100
		self.percent_back_start = 100
		self.percent_JA_start = 100
		self.percent_fore_step = 2
		self.percent_back_step = 2
		self.percent_JA_step = 2
		self.epoch_start = 30
		self.EXP_NAME = 'CVC_DW_HieraSeg'
		self.EPS = 0.
		print('CBloss_weight:', self.CBloss_weight)
		print('normal_dim0:', self.normal_dim0)
		print('polyp_dim0:', self.polyp_dim0)
		print('normal_dim1:', self.normal_dim1)
		print('polyp_dim1:', self.polyp_dim1)
		print('normal_dim2:', self.normal_dim2)
		print('polyp_dim2:', self.polyp_dim2)
		print('percent_fore_start:', self.percent_fore_start)
		print('percent_back_start:', self.percent_back_start)
		print('percent_JA_start:', self.percent_JA_start)
		print('percent_fore_step:', self.percent_fore_step)
		print('percent_back_step:', self.percent_back_step)
		print('percent_JA_step:', self.percent_JA_step)
		print('epoch_start:', self.epoch_start)
		print('EPS:', self.EPS)

		self.DATA_NAME = 'CVC'
		self.DATA_AUG = False
		self.DATA_WORKERS = 2
		self.DATA_RESCALE = 256
		self.DATA_RANDOMCROP = 384
		self.DATA_RANDOMROTATION = 180
		self.DATA_RANDOMSCALE = 1.25
		self.DATA_RANDOM_H = 10
		self.DATA_RANDOM_S = 10
		self.DATA_RANDOM_V = 10
		self.DATA_RANDOMFLIP = 0.5
			
		self.MODEL_NAME = 'deeplabv3plus'
		self.MODEL_BACKBONE = 'res101_atrous'
		self.MODEL_OUTPUT_STRIDE = 16
		self.MODEL_ASPP_OUTDIM = 256
		self.MODEL_SHORTCUT_DIM = 48
		self.MODEL_SHORTCUT_KERNEL = 1
		self.MODEL_NUM_CLASSES = 2
		self.MODEL_SAVE_DIR = os.path.join(self.ROOT_DIR,'model',self.EXP_NAME)

		self.TRAIN_LR = 0.001
		print('TRAIN_LR:', self.TRAIN_LR)
		self.TRAIN_LR_GAMMA = 0.1
		self.TRAIN_MOMENTUM = 0.9
		self.TRAIN_WEIGHT_DECAY = 0.00004
		self.TRAIN_BN_MOM = 0.0003
		self.TRAIN_POWER = 0.9 #0.9
		self.TRAIN_GPUS = 1
		self.TRAIN_BATCHES = 16
		self.TRAIN_SHUFFLE = True
		self.TRAIN_MINEPOCH = 0	
		self.TRAIN_EPOCHS = 500
		self.TRAIN_LOSS_LAMBDA = 0
		self.TRAIN_TBLOG = True
		self.TRAIN_CKPT = os.path.join(self.ROOT_DIR,'/home/xiaoqiguo2/Threshold/model/deeplabv3plus_res101_atrous_VOC2012_epoch46_all.pth')
	#	self.TRAIN_CKPT = os.path.join(self.ROOT_DIR,'./model/WCE125_lr001_256_256threshold_margin/deeplabv3plus_res101_atrous_WCE_epoch500_all.pth')

		self.LOG_DIR = os.path.join(self.ROOT_DIR,'log',self.EXP_NAME)

		self.TEST_MULTISCALE = [1.0]#[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
		self.TEST_FLIP = False#True
		self.TEST_CKPT = os.path.join(self.ROOT_DIR,'./model/CVC_DWNet_MultiFC/model-best-deeplabv3plus_res101_atrous_CVC_epoch465_jac81.036.pth')
		self.TEST_GPUS = 1
		self.TEST_BATCHES = 32	

		self.__check()
		self.__add_path(os.path.join(self.ROOT_DIR, 'lib'))
		
	def __check(self):
		if not torch.cuda.is_available():
			raise ValueError('config.py: cuda is not avalable')
		if self.TRAIN_GPUS == 0:
			raise ValueError('config.py: the number of GPU is 0')
		#if self.TRAIN_GPUS != torch.cuda.device_count():
		#	raise ValueError('config.py: GPU number is not matched')
		if not os.path.isdir(self.LOG_DIR):
			os.makedirs(self.LOG_DIR)
		if not os.path.isdir(self.MODEL_SAVE_DIR):
			os.makedirs(self.MODEL_SAVE_DIR)

	def __add_path(self, path):
		if path not in sys.path:
			sys.path.insert(0, path)

cfg = Configuration() 	
