# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
import os

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from net.sync_batchnorm.replicate import patch_replication_callback

from torch.utils.data import DataLoader

def Multi2Binary(inputs, normal_dim=4, polyp_dim=2):
#    inputs = nn.Softmax(dim=1)(inputs)
    normal_prob = torch.sum(inputs[:,:normal_dim,:,:], dim=1, keepdim=True)
    polyp_prob = torch.sum(inputs[:,normal_dim:,:,:], dim=1, keepdim=True)
    prob = torch.cat([normal_prob, polyp_prob],1)
    return prob

def test_net():
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test')
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TEST_BATCHES, 
				shuffle=False, 
				num_workers=cfg.DATA_WORKERS)
	
	net = generate_net(cfg)
	print('net initialize')
	if cfg.TEST_CKPT is None:
		raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
	

	print('Use %d GPU'%cfg.TEST_GPUS)
	device = torch.device('cuda')
	if cfg.TEST_GPUS > 1:
		net = nn.DataParallel(net)
		patch_replication_callback(net)
	net.to(device)

	print('start loading model %s'%cfg.TEST_CKPT)
	model_dict = torch.load(cfg.TEST_CKPT,map_location=device)
	net.load_state_dict(model_dict)
	
#	for k, v in model_dict.items():
#		print(k, v)

	Acc_array = 0.
	Prec_array = 0.
	Spe_array = 0.
	Rec_array = 0.
	IoU_array = 0.
	Dice_array = 0.
	HD_array = 0.
	sample_num = 0.
	result_list = []
	CEloss_list = []
	JAloss_list = []
	Label_list = []
	net.eval()
	folder_path = os.path.join(cfg.ROOT_DIR,'data','results','CVC','features')
	with torch.no_grad():
		for i_batch, sample_batched in enumerate(dataloader):
			name_batched = sample_batched['name']
			row_batched = sample_batched['row']
			col_batched = sample_batched['col']

			[batch, channel, height, width] = sample_batched['image'].size()
			multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).cuda()
			for rate in cfg.TEST_MULTISCALE:
				inputs_batched = sample_batched['image_%f'%rate]
				inputs_batched = inputs_batched.cuda()
				_, predicts0, predicts1, predicts2 = net(inputs_batched, FC_pyramid = True)
				predicts0 = Multi2Binary(predicts0.cuda(), normal_dim=cfg.normal_dim0, polyp_dim=cfg.polyp_dim0)
				predicts1 = Multi2Binary(predicts1.cuda(), normal_dim=cfg.normal_dim1, polyp_dim=cfg.polyp_dim1)
				predicts2 = Multi2Binary(predicts2.cuda(), normal_dim=cfg.normal_dim2, polyp_dim=cfg.polyp_dim2)
			#	predicts2 = predicts2.cuda()
				predicts = (predicts0 + predicts1 + predicts2)/3.
				predicts_batched = predicts.clone()
				del predicts0
				del predicts1
				del predicts2
			
				#predicts_batched = F.interpolate(predicts_batched, size=None, scale_factor=1/rate, mode='bilinear', align_corners=True)
				multi_avg = multi_avg + predicts_batched
				del predicts_batched			
			multi_avg = multi_avg / len(cfg.TEST_MULTISCALE)
			result = torch.argmax(multi_avg, dim=1).cpu().numpy().astype(np.uint8)
			labels_batched = sample_batched['segmentation']

			for i in range(batch):
				row = row_batched[i]
				col = col_batched[i]
				p = result[i,:,:]					
				l = labels_batched[i,:,:]
			#	ff = features_batched[i,:,:,:]
				#p = cv2.resize(p, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				#l = cv2.resize(l, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				predict = np.int32(p)
				gt = np.int32(l)
				cal = gt<255
				mask = (predict==gt) * cal 
				TP = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
				TN = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
				P = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
				T = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)  

				P = np.sum((predict==1)).astype(np.float64)
				T = np.sum((gt==1)).astype(np.float64)
				TP = np.sum((gt==1)*(predict==1)).astype(np.float64)
				TN = np.sum((gt==0)*(predict==0)).astype(np.float64)

				Acc = (TP+TN)/(T+P-TP+TN)
				Prec = TP/(P+10e-6)
				Spe = TN/(P-TP+TN)
				Rec = TP/T
				DICE = 2*TP/(T+P)
				IoU = TP/(T+P-TP)
			#	HD = max(directed_hausdorff(predict, gt)[0], directed_hausdorff(predict, gt)[0])
			#	HD = 2*Prec*Rec/(Rec+Prec+1e-10)
				beta = 2
				HD = Rec*Prec*(1+beta**2)/(Rec+beta**2*Prec+1e-10)
				Acc_array += Acc
				Prec_array += Prec
				Spe_array += Spe
				Rec_array += Rec
				Dice_array += DICE
				IoU_array += IoU
				HD_array += HD
				sample_num += 1
				#p = cv2.resize(p, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				result_list.append({'predict':np.uint8(p*255), 'label':np.uint8(l*255), 'IoU':IoU, 'name':name_batched[i]})
				
			#	if not os.path.exists(folder_path):
			#		os.makedirs(folder_path)
			#	new_name = name_batched[i].split('.')[0]+'.npy'
			#	file_path = os.path.join(folder_path, '%s'%new_name)
			#	np.save(file_path, ff) 

		Acc_score = Acc_array*100/sample_num
		Prec_score = Prec_array*100/sample_num
		Spe_score = Spe_array*100/sample_num
		Rec_score = Rec_array*100/sample_num
		Dice_score = Dice_array*100/sample_num
		IoUP = IoU_array*100/sample_num
		HD_score = HD_array*100/sample_num
		print('%10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%\n'%('Acc',Acc_score,'Sen',Rec_score,'Spe',Spe_score,'Prec',Prec_score,'Dice',Dice_score,'Jac',IoUP,'F2',HD_score))
	dataset.save_result(result_list, cfg.MODEL_NAME)
#	dataset.do_python_eval(cfg.MODEL_NAME)
	print('Test finished')

if __name__ == '__main__':
	test_net()


