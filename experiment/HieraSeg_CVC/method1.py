# ----------------------------------------
# Written by Xiaoqing Guo
# ----------------------------------------

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np
import torch.nn.functional as F
import cv2

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from net.loss import MaskCrossEntropyLoss, MaskBCELoss, MaskBCEWithLogitsLoss
from net.sync_batchnorm.replicate import patch_replication_callback
from scipy.spatial.distance import directed_hausdorff

def Jaccard_loss_cal(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)[:,1,:,:].cuda()
    true_1_hot = true_1_hot.type(logits.type())[:,1,:,:].cuda()
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dim=(1,2))
    cardinality = torch.sum(probas + true_1_hot, dim=(1,2))
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps))
    return (1. - jacc_loss).sum()

def Multi2Binary(inputs, normal_dim=4, polyp_dim=2):
#    inputs = nn.Softmax(dim=1)(inputs)
    normal_prob = torch.sum(inputs[:,:normal_dim,:,:], dim=1, keepdim=True)
    polyp_prob = torch.sum(inputs[:,normal_dim:,:,:], dim=1, keepdim=True)
    prob = torch.cat([normal_prob, polyp_prob],1)
    return prob

def MulCls_Label_Gen(inputs, target, normal_dim=4, polyp_dim=2):
	Normal_inputs = inputs[:, :normal_dim, :, :]	
	Polyp_inputs = inputs[:, normal_dim:, :, :]
	Normal_ind = torch.argmax(Normal_inputs, dim=1)
	Polyp_ind = torch.argmax(Polyp_inputs, dim=1) + normal_dim

	labels_ind = target > 0.5
	multi_label = torch.where(labels_ind, Polyp_ind, Normal_ind)
	return multi_label

def WeightIntraLoss(W, dim = 4):
	W_norm = torch.norm(W, p=2, dim=1, keepdim=True) 
	W = torch.div(W, W_norm) 	
	loss = torch.mm(W, torch.transpose(W, 0, 1))

	Eye = torch.eye(dim).cuda()
	Zero_like = torch.zeros(dim,dim).cuda()
	loss = torch.where(Eye>0., Zero_like, loss)
	loss = loss.sum() / (dim * (dim-1))
	return loss

def WeightInterLoss(WN, WP):
	WN_mean = torch.mean(WN, dim=0)
	WP_mean = torch.mean(WP, dim=0)
	loss = (torch.matmul(WN_mean, WP_mean) / (torch.norm(WN_mean) * torch.norm(WP_mean)) + 1) 
	return loss

def WeightLoss(Weight, normal_dim=4, polyp_dim=2):
	Weight = torch.squeeze(Weight)

	Normal_Weight = Weight[:normal_dim,:]	
	Polyp_Weight = Weight[normal_dim:,:]

	Normal_intra_loss = WeightIntraLoss(Normal_Weight, dim = normal_dim) + 1
	Polyp_intra_loss = WeightIntraLoss(Polyp_Weight, dim = polyp_dim) + 1

	loss_intra = Normal_intra_loss + Polyp_intra_loss
	loss_inter = WeightInterLoss(Normal_Weight, Polyp_Weight)
	loss = 0.5 * loss_intra + loss_inter
	return loss

def make_one_hot(labels):
	target = torch.eye(cfg.MODEL_NUM_CLASSES)[labels]
	gt_1_hot = target.permute(0, 3, 1, 2).float().cuda()
	return gt_1_hot

def Label_smoothing(labels, class_number = cfg.MODEL_NUM_CLASSES, eps = cfg.EPS):
	one_hot = make_one_hot(labels, class_number)
	labels_ind = one_hot>0.5
	label_smoothing = torch.where(labels_ind, one_hot - eps*(class_number-1), one_hot + eps)
	return label_smoothing

def model_snapshot(model, new_file=None, old_file=None):
    if os.path.exists(old_file) is True:
        os.remove(old_file) 
    torch.save(model, new_file)
    print('      %s has been saved\n'%new_file)

def train_one_batch(inputs_batched, labels_batched, net, seg_optimizer, epoch):
	Predicts, predicts_batched, multiclass_label = {}, {}, {}
	net.train() 
	criterion = nn.CrossEntropyLoss(ignore_index=255)
	features_batch, predicts_batched["{0}".format(0)], predicts_batched["{0}".format(1)], predicts_batched["{0}".format(2)] = net(inputs_batched, labels_batched, FC_pyramid = True, phase = 'train')
	feature = torch.from_numpy(features_batch.cpu().data.numpy()).cuda()
	Predicts["{0}".format(0)] = Multi2Binary(predicts_batched["{0}".format(0)].cuda(), normal_dim=cfg.normal_dim0, polyp_dim=cfg.polyp_dim0)

	predicts = Predicts["{0}".format(0)] 
	result = torch.argmax(predicts, dim=1).cpu().numpy().astype(np.uint8)
	CE_loss = 0.
	JA_loss = 0.
	for i in range(1):
		JA_loss += Jaccard_loss_cal(labels_batched, Predicts["{0}".format(i)], eps=1e-7)
		CE_loss += criterion(Predicts["{0}".format(i)], labels_batched)

	seg_loss = (CE_loss + JA_loss)/3.
	seg_optimizer.zero_grad()
	seg_loss.backward()
	seg_optimizer.step()
	loss_weight0 = CE_loss
	loss_weight1 = CE_loss

	return CE_loss, JA_loss, loss_weight0, loss_weight1, result

def train_net():
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train', cfg.DATA_AUG)
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TRAIN_BATCHES, 
				shuffle=cfg.TRAIN_SHUFFLE, 
				num_workers=cfg.DATA_WORKERS,
				drop_last=True)

	test_dataset = generate_dataset(cfg.DATA_NAME, cfg, 'test')
	test_dataloader = DataLoader(test_dataset, 
				batch_size=cfg.TEST_BATCHES, 
				shuffle=False, 
				num_workers=cfg.DATA_WORKERS)

	net = generate_net(cfg)

	print('Use %d GPU'%cfg.TRAIN_GPUS)
	device = torch.device(0)
	net = nn.DataParallel(net)
	patch_replication_callback(net)
	net.to(device)		

	if cfg.TRAIN_CKPT:
		pretrained_dict = torch.load(cfg.TRAIN_CKPT)
		net_dict = net.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
		net_dict.update(pretrained_dict)
		net.load_state_dict(net_dict)

	net = net.module

	segment_dict = []
	hierafc_dict = []
	backbone_dict = []
	for i, para in enumerate(net.parameters()):
		if i < 36:
			segment_dict.append(para)
		elif i < 39:
			hierafc_dict.append(para)
		else:
			backbone_dict.append(para)

	seg_optimizer = optim.SGD(
		params = [
            {'params': backbone_dict, 'lr': cfg.TRAIN_LR},
            {'params': hierafc_dict, 'lr': 30*cfg.TRAIN_LR},
            {'params': segment_dict, 'lr': 10*cfg.TRAIN_LR}
        ],
		momentum=cfg.TRAIN_MOMENTUM,
		weight_decay=cfg.TRAIN_WEIGHT_DECAY)

	itr = cfg.TRAIN_MINEPOCH * len(dataloader)
	max_itr = cfg.TRAIN_EPOCHS * len(dataloader)
	best_jacc = 0.
	best_epoch = 0
	last_jacc = 0.
	for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):
		#### start training now
		IoU_array = 0.
		sample_num = 0.
		running_CE_loss = 0.0
		running_JA_loss = 0.0 
		running_weight0_loss = 0.0 
		running_weight1_loss = 0.0 
		dataset_list = []
		 
        #########################################################
        ########### give lambda && decay coefficient ############
        #########################################################        
		for i_batch, sample_batched in enumerate(dataloader):
			now_lr = adjust_lr(seg_optimizer, itr, max_itr)
			inputs_batched, labels_batched = sample_batched['image'], sample_batched['segmentation']
			inputs_batched = inputs_batched.cuda()
			labels_batched = labels_batched.long().cuda()
			
			CE_loss, JA_loss, loss_weight0, loss_weight1, result = train_one_batch(inputs_batched, labels_batched, net, seg_optimizer, epoch)
			
			running_CE_loss += CE_loss.item()
			running_JA_loss += JA_loss.item()
			running_weight0_loss += loss_weight0.item()
			running_weight1_loss += loss_weight1.item()		
			itr += 1

		i_batch = i_batch + 1
		print('epoch:%d/%d\tCE loss:%g\tJA loss:%g\tWeight0 loss:%g\tWeight1 loss:%g' % (epoch, cfg.TRAIN_EPOCHS, 
		            running_CE_loss/i_batch, running_JA_loss/i_batch, running_weight0_loss/i_batch, running_weight1_loss/i_batch))
		
		#### start testing now
		if (epoch) % 1 == 0:
			IoUP = test_one_epoch(test_dataset, test_dataloader, net, epoch)
			if IoUP > best_jacc:
				model_snapshot(net.state_dict(), new_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_jac%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch,IoUP)),
                		old_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_jac%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,best_epoch,best_jacc)))
				best_jacc = IoUP
				best_epoch = epoch

def adjust_lr(optimizer, itr, max_itr):
	now_lr = cfg.TRAIN_LR * (1 - (itr/(max_itr+1)) ** cfg.TRAIN_POWER)
#	now_lr = np.max([cfg.TRAIN_LR*0.1, now_lr])
	optimizer.param_groups[0]['lr'] = now_lr
	optimizer.param_groups[1]['lr'] = 30*now_lr
	optimizer.param_groups[2]['lr'] = 10*now_lr
	return now_lr

def get_params(model, key):
	for m in model.named_modules():
		if key == '1x':
			if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
				for p in m[1].parameters():
					yield p
		elif key == '10x':
			if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
				for p in m[1].parameters():
					yield p

def test_one_itr(result, labels_batched):
	IoUarray = 0.
	samplenum = 0.
	label = labels_batched.cpu().numpy()
	batch = result.shape[0]
	del labels_batched			
	for i in range(batch):
		p = result[i,:,:]					
		l = label[i,:,:]
		predict = np.int32(p)
		gt = np.int32(l)
		TP = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
		P = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)
		T = np.zeros((cfg.MODEL_NUM_CLASSES), np.uint64)  

		P = np.sum((predict==1)).astype(np.float64)
		T = np.sum((gt==1)).astype(np.float64)
		TP = np.sum((gt==1)*(predict==1)).astype(np.float64)

		IoU = TP/(T+P-TP+10e-5)
		IoUarray += IoU
		samplenum += 1

	return IoUarray, samplenum

def test_one_epoch(dataset, DATAloader, net, epoch):
	#### start testing now
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
	with torch.no_grad():
		for i_batch, sample_batched in enumerate(DATAloader):
			name_batched = sample_batched['name']
			row_batched = sample_batched['row']
			col_batched = sample_batched['col']

			[batch, channel, height, width] = sample_batched['image'].size()
			multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).cuda()
			labels_batched = sample_batched['segmentation'].cpu().numpy()
			for rate in cfg.TEST_MULTISCALE:
				inputs_batched = sample_batched['image_%f'%rate]
				inputs_batched = inputs_batched.cuda()
				_, predicts0, predicts1, predicts2 = net(inputs_batched, FC_pyramid = True)
				predicts0 = Multi2Binary(predicts0.cuda(), normal_dim=cfg.normal_dim0, polyp_dim=cfg.polyp_dim0)
				predicts = predicts0
				predicts_batched = predicts.clone()
				del predicts0
			
				predicts_batched = F.interpolate(predicts_batched, size=None, scale_factor=1/rate, mode='bilinear', align_corners=True)
				multi_avg = multi_avg + predicts_batched
				del predicts_batched			
			multi_avg = multi_avg / len(cfg.TEST_MULTISCALE)
			result = torch.argmax(multi_avg, dim=1).cpu().numpy().astype(np.uint8)

			for i in range(batch):
				row = row_batched[i]
				col = col_batched[i]
				p = result[i,:,:]					
				l = labels_batched[i,:,:]
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

		Acc_score = Acc_array*100/sample_num
		Prec_score = Prec_array*100/sample_num
		Spe_score = Spe_array*100/sample_num
		Rec_score = Rec_array*100/sample_num
		Dice_score = Dice_array*100/sample_num
		IoUP = IoU_array*100/sample_num
		HD_score = HD_array*100/sample_num
		print('%10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%\n'%('Acc',Acc_score,'Sen',Rec_score,'Spe',Spe_score,'Prec',Prec_score,'Dice',Dice_score,'Jac',IoUP,'F2',HD_score))
		if epoch % 50 == 0:
			dataset.save_result_train(result_list, cfg.MODEL_NAME)

		return IoUP

if __name__ == '__main__':
	train_net()