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
    return (1. - jacc_loss)

def SoftCrossEntropy(inputs, target, reduction=True, eps=1e-6):
    inputs = nn.Softmax(dim=1)(inputs)
    log_likelihood = -torch.log(inputs+eps)
    if reduction:
        loss = torch.mean(torch.sum(torch.mul(log_likelihood, target), dim=1))
    else:
        loss = torch.sum(torch.mul(log_likelihood, target), dim=1)
    return loss

def make_one_hot(labels, class_number=2):
	target = torch.eye(class_number)[labels]
	gt_1_hot = target.permute(0, 3, 1, 2).float().cuda()
	return gt_1_hot

def Label_smoothing(labels, class_number = cfg.MODEL_NUM_CLASSES, eps = cfg.EPS):
	one_hot = make_one_hot(labels, class_number)
	labels_ind = one_hot>0.5
	label_smoothing = torch.where(labels_ind, one_hot - eps*(class_number-1), one_hot + eps)
	return label_smoothing

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

def CE_SPLweight_gen(loss, labels, lambda_fore, lambda_back, eps=1e-10):
	labels_ind = labels>0.
	loss_class1 = torch.where(labels_ind, loss, torch.ones_like(loss)*1000.)
	loss_class0 = torch.where(labels_ind, torch.ones_like(loss)*1000., loss)

	ClipV_weights_tensor_cpu1 = torch.where(loss_class1<=lambda_fore, torch.ones_like(loss), torch.zeros_like(loss)).cpu().data.numpy()
	ClipV_weights_tensor_cpu0 = torch.where(loss_class0<=lambda_back, torch.ones_like(loss), torch.zeros_like(loss)).cpu().data.numpy()	
	ClipV_weights_tensor = ClipV_weights_tensor_cpu1 + ClipV_weights_tensor_cpu0
	return ClipV_weights_tensor

def JA_SPLweight_gen(loss, lambda_JA, eps=1e-10):
	ClipV_weights_JA_cpu = torch.where(loss<=lambda_JA, torch.ones_like(loss), torch.zeros_like(loss)).cpu().data.numpy()
	return ClipV_weights_JA_cpu

def CE_SPLloss_gen(loss, ClipV_weights_tensor1, ClipV_weights_tensor0, eps=1e-10):
	if torch.sum(ClipV_weights_tensor1) == 0:
		loss_class1 = torch.sum(loss*ClipV_weights_tensor1)
	else:
		loss_class1 = torch.sum(loss*ClipV_weights_tensor1)/torch.sum(ClipV_weights_tensor1)

	if torch.sum(ClipV_weights_tensor0) == 0:
		loss_class0 = torch.sum(loss*ClipV_weights_tensor0)
	else:
		loss_class0 = torch.sum(loss*ClipV_weights_tensor0)/torch.sum(ClipV_weights_tensor0)

	CE_SPLloss = loss_class1 + loss_class0
	return loss_class1, loss_class0

def JA_SPLloss_gen(loss, ClipV_weights_JA, eps=1e-10):
	if torch.sum(ClipV_weights_JA).item() == 0:
		SPL_loss = torch.sum(loss*ClipV_weights_JA)
	else:
		SPL_loss = torch.sum(loss*ClipV_weights_JA)
#		SPL_loss = torch.sum(loss*ClipV_weights_JA)/torch.sum(ClipV_weights_JA)
	return SPL_loss

def CE_Lambda_gen(loss, labels, percent_fore, percent_back, eps=1e-5):
	loss = loss.cpu().data.numpy()
	labels = labels.cpu().data.numpy()
	len_class1 = np.sum(labels)
	len_class0 = np.sum(np.ones_like(labels)) - np.sum(labels)
	labels_ind = labels>0.5

	loss_class1 = np.where(labels_ind, loss, np.ones_like(loss)*10e9)
	loss_class0 = np.where(labels_ind, np.ones_like(loss)*10e9, loss)
	loss_class1_sorted = np.sort(loss_class1, axis=None) 
	loss_class0_sorted = np.sort(loss_class0, axis=None) 

	index1 = np.int32(len_class1 * percent_fore)
	index0 = np.int32(len_class0 * percent_back)
	lambda_class1 = loss_class1_sorted[index1-1] + eps
	lambda_class0 = loss_class0_sorted[index0-1] + eps
	return lambda_class1, lambda_class0

def JA_Lambda_gen(loss, percent):
	loss = loss.cpu().data.numpy()
	len_loss = np.sum(np.ones_like(loss)) 
	loss_sorted = np.sort(loss, axis=None) 
	index = np.int32(len_loss * percent)
	lambda_JA = loss_sorted[index-1]
	return lambda_JA

def model_snapshot_noW(model, new_file=None, old_file=None):
    if os.path.exists(old_file) is True:
        os.remove(old_file) 
    torch.save(model, new_file)
    print('      %s has been saved\n'%new_file)

def model_snapshot(model, Wmodel, new_file=None, old_file=None, new_Wfile=None, old_Wfile=None):
    if os.path.exists(old_file) is True:
        os.remove(old_file) 
    torch.save(model, new_file)

    if os.path.exists(old_Wfile) is True:
        os.remove(old_Wfile) 
    torch.save(Wmodel, new_Wfile)
    print('      %s has been saved\n'%new_file)

def train_one_batch(train_list):
	inputs_batched = train_list['inputs_batched']
	labels_batched = train_list['labels_batched']
	percent_fore = train_list['percent_fore']
	percent_back = train_list['percent_back']
	percent_JA = train_list['percent_JA']
	Predicts, predicts_batched, multiclass_label = {}, {}, {}
	
	train_list['net'].train() 
	features_batch, predicts_batched["{0}".format(0)], predicts_batched["{0}".format(1)], predicts_batched["{0}".format(2)] = train_list['net'](inputs_batched, labels_batched, FC_pyramid = True, phase = 'train')
	feature = torch.from_numpy(features_batch.cpu().data.numpy()).cuda()
	labels_smoothing = Label_smoothing(labels_batched)
	Predicts["{0}".format(0)] = Multi2Binary(predicts_batched["{0}".format(0)].cuda(), normal_dim=cfg.normal_dim0, polyp_dim=cfg.polyp_dim0)
	Predicts["{0}".format(1)] = Multi2Binary(predicts_batched["{0}".format(1)].cuda(), normal_dim=cfg.normal_dim1, polyp_dim=cfg.polyp_dim1)
	Predicts["{0}".format(2)] = Multi2Binary(predicts_batched["{0}".format(2)].cuda(), normal_dim=cfg.normal_dim2, polyp_dim=cfg.polyp_dim2)
	multiclass_label["{0}".format(0)] = MulCls_Label_Gen(predicts_batched["{0}".format(0)], labels_batched, normal_dim=cfg.normal_dim0, polyp_dim=cfg.polyp_dim0)
	multiclass_label["{0}".format(1)] = MulCls_Label_Gen(predicts_batched["{0}".format(1)], labels_batched, normal_dim=cfg.normal_dim1, polyp_dim=cfg.polyp_dim1)
	multiclass_label["{0}".format(2)] = MulCls_Label_Gen(predicts_batched["{0}".format(2)], labels_batched, normal_dim=cfg.normal_dim2, polyp_dim=cfg.polyp_dim2)

	MSE = torch.nn.MSELoss(reduce=True, reduction='mean')
	criterion = nn.CrossEntropyLoss(reduce=False, ignore_index=255)
	predicts = (Predicts["{0}".format(0)] + Predicts["{0}".format(1)] + Predicts["{0}".format(2)])/3.
	result = torch.argmax(predicts, dim=1).cpu().numpy().astype(np.uint8)

	CE_loss, JA_loss, pix_loss, img_loss, pix_weight, img_weight, weight_loss = {}, {}, {}, {}, {}, {}, {}
	pix_weight_cpu, img_weight_cpu = {}, {}
	for i in range(3):
		CE_loss["{0}".format(i)] = SoftCrossEntropy(Predicts["{0}".format(i)], labels_smoothing, reduction=False)
		if train_list['epoch'] >= cfg.epoch_start:
			cls_num = predicts_batched["{0}".format(i)].shape[1]
			multi_labels_smoothing = Label_smoothing(multiclass_label["{0}".format(i)], class_number = cls_num, eps = cfg.EPS/(cls_num-1.))
			multi_CE_loss = SoftCrossEntropy(predicts_batched["{0}".format(i)], multi_labels_smoothing, reduction=False)
			CE_loss["{0}".format(i)] += multi_CE_loss
		lambda_fore, lambda_back = CE_Lambda_gen(CE_loss["{0}".format(i)], labels_batched, percent_fore, percent_back)
		lambda_CE = lambda_fore * labels_batched + lambda_back * (1 - labels_batched)
		percent_CE = percent_fore * labels_batched + percent_back * (1 - labels_batched)
		ClipV_weights_CE = CE_SPLweight_gen(CE_loss["{0}".format(i)], labels_batched, lambda_fore, lambda_back, eps=1e-10)

		JA_loss["{0}".format(i)] = Jaccard_loss_cal(labels_batched, Predicts["{0}".format(i)], eps=1e-7)
		lambda_JA = JA_Lambda_gen(JA_loss["{0}".format(i)], percent_JA)
		ClipV_weights_JA = JA_SPLweight_gen(JA_loss["{0}".format(i)], lambda_JA, eps=1e-10)
		
		img_loss["{0}".format(i)] = torch.from_numpy(JA_loss["{0}".format(i)].cpu().data.numpy() - lambda_JA).cuda()
		pix_loss["{0}".format(i)] = torch.from_numpy((CE_loss["{0}".format(i)] - lambda_CE).cpu().data.numpy()).cuda()
		img_w, pix_w = train_list['WeightNet'+str(i)](img_loss["{0}".format(i)], percent_JA, pix_loss["{0}".format(i)], percent_CE, feature, labels_batched.cuda())
		
		img_weight["{0}".format(i)] = torch.squeeze(img_w).cuda()
		pix_weight["{0}".format(i)] = torch.squeeze(pix_w).cuda()
		img_weight_cpu["{0}".format(i)] = torch.from_numpy(img_weight["{0}".format(i)].cpu().data.numpy()).cuda()
		pix_weight_cpu["{0}".format(i)] = torch.from_numpy(pix_weight["{0}".format(i)].cpu().data.numpy()).cuda()
		img_weight_loss = MSE(img_weight["{0}".format(i)], torch.from_numpy(ClipV_weights_JA).cuda())
		pix_weight_loss = MSE(pix_weight["{0}".format(i)], torch.from_numpy(ClipV_weights_CE).cuda())
		fore_pixels = torch.mean(torch.sum(pix_weight["{0}".format(i)].mul(labels_batched), dim=(1,2)))
		back_pixels = torch.mean(torch.sum(pix_weight["{0}".format(i)].mul(1.-labels_batched), dim=(1,2)))
		class_imbalance_loss = (1. - fore_pixels/back_pixels)**2

		'''weight_loss = img_weight_loss + pix_weight_loss + cfg.CBloss_weight * class_imbalance_loss
		train_list['wei_optimizer'+str(i)].zero_grad()
		weight_loss.backward()
		train_list['wei_optimizer'+str(i)].step()'''

		weight_loss["{0}".format(i)] = img_weight_loss + pix_weight_loss + cfg.CBloss_weight * class_imbalance_loss
	if train_list['epoch'] >= cfg.epoch_start:
		for i in range(3):
			for j in range(3):
				if i != j:
					weight_loss["{0}".format(i)] += 0.5 * MSE(img_weight["{0}".format(i)], img_weight_cpu["{0}".format(j)])
					weight_loss["{0}".format(i)] += 0.5 * MSE(pix_weight["{0}".format(i)], pix_weight_cpu["{0}".format(j)])

	for i in range(3):
		train_list['wei_optimizer'+str(i)].zero_grad()
		weight_loss["{0}".format(i)].backward()
		train_list['wei_optimizer'+str(i)].step()

	CE_SPLloss_fore = 0.
	CE_SPLloss_back = 0.
	JA_SPLloss = 0.
	for i in range(3):
		# update the prediction of WeightNet
		with torch.no_grad():
			img_weight, pix_weight = train_list['WeightNet'+str(i)](img_loss["{0}".format(i)], percent_JA, pix_loss["{0}".format(i)], percent_CE, feature, labels_batched.cuda())
			img_weight = torch.from_numpy(img_weight.cpu().data.numpy()).cuda()
			pix_weight = torch.from_numpy(pix_weight.cpu().data.numpy()).cuda()
			img_weight = torch.squeeze(img_weight)
			pix_weight = torch.squeeze(pix_weight)
		
		# obtain the loss of Segmentation Network  
		fore, back = CE_SPLloss_gen(CE_loss["{0}".format(i)], pix_weight.mul(labels_batched), pix_weight.mul(1.-labels_batched), eps=1e-10)
		CE_SPLloss_fore += fore
		CE_SPLloss_back += back
		JA_SPLloss += JA_SPLloss_gen(JA_loss["{0}".format(i)], img_weight, eps=1e-10)
		
	SPL_loss = (CE_SPLloss_fore + CE_SPLloss_back + JA_SPLloss)/3.

	# obtain the weight loss in classifier1
	weight0 = list(train_list['net'].cls_conv_0.parameters())[0]
	loss_weight0 = WeightLoss(weight0, normal_dim=cfg.normal_dim0, polyp_dim=cfg.polyp_dim0)

	weight1 = list(train_list['net'].cls_conv_1.parameters())[0]
	loss_weight1 = WeightLoss(weight1, normal_dim=cfg.normal_dim1, polyp_dim=cfg.polyp_dim1)

#	weight2 = list(train_list['net'].cls_conv_2.parameters())[0]
#	loss_weight2 = WeightLoss(weight2, normal_dim=cfg.normal_dim2, polyp_dim=cfg.polyp_dim2)

	loss_weight = 0.2 * (loss_weight0 + loss_weight1).cuda()

	seg_loss = SPL_loss + loss_weight
	train_list['seg_optimizer'].zero_grad()
	seg_loss.backward()
	train_list['seg_optimizer'].step()

	return img_weight_loss, pix_weight_loss, class_imbalance_loss, CE_SPLloss_fore, CE_SPLloss_back, JA_SPLloss, img_weight, pix_weight, result

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
	WeightNet0 = generate_net(cfg, is_learning = True)
	WeightNet1 = generate_net(cfg, is_learning = True)
	WeightNet2 = generate_net(cfg, is_learning = True)

	print('Use %d GPU'%cfg.TRAIN_GPUS)
	device = torch.device(0)
	net = nn.DataParallel(net)
	patch_replication_callback(net)
	net.to(device)		
	WeightNet0.to(device)		
	WeightNet1.to(device)		
	WeightNet2.to(device)	

	if cfg.TRAIN_CKPT:
		pretrained_dict = torch.load(cfg.TRAIN_CKPT)
		net_dict = net.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
		net_dict.update(pretrained_dict)
		net.load_state_dict(net_dict)

		'''WeightNet_pretrained_dict = torch.load('/home/xiaoqiguo2/LearnSPL/model/CVC_DWNet424_CBW10_EANetJAsum_labelEps1/model-best-deeplabv3plus_res101_atrous_CVC_epoch424_jac80.824.pth')
		WeightNet_dict = WeightNet.state_dict()
		WeightNet_pretrained_dict = {k: v for k, v in WeightNet_pretrained_dict.items() if (k in WeightNet_dict) and (v.shape==WeightNet_dict[k].shape)}
		WeightNet_dict.update(WeightNet_pretrained_dict)
		WeightNet.load_state_dict(WeightNet_dict)'''

	net = net.module

	weight_dict0 = []
	weight_dict1 = []
	weight_dict2 = []
	segment_dict = []
	hierafc_dict = []
	backbone_dict = []

#	for i, para in enumerate(net.named_parameters()):
#		(name, param) = para
#		print(i, name)	
#	for k, v in pretrained_dict.items():
#		print(k)	
#	print(i, name)[0]
	
	for i, para in enumerate(net.parameters()):
		if i < 36:
			segment_dict.append(para)
		elif i < 39:
			hierafc_dict.append(para)
		else:
			backbone_dict.append(para)

	wei_optimizer0 = optim.SGD(WeightNet0.parameters(), lr=10*cfg.TRAIN_LR, momentum=cfg.TRAIN_MOMENTUM, weight_decay=cfg.TRAIN_WEIGHT_DECAY)
	wei_optimizer1 = optim.SGD(WeightNet1.parameters(), lr=10*cfg.TRAIN_LR, momentum=cfg.TRAIN_MOMENTUM, weight_decay=cfg.TRAIN_WEIGHT_DECAY)
	wei_optimizer2 = optim.SGD(WeightNet2.parameters(), lr=10*cfg.TRAIN_LR, momentum=cfg.TRAIN_MOMENTUM, weight_decay=cfg.TRAIN_WEIGHT_DECAY)
	seg_optimizer = optim.SGD(
		params = [
            {'params': backbone_dict, 'lr': cfg.TRAIN_LR},
            {'params': hierafc_dict, 'lr': 30*cfg.TRAIN_LR},
            {'params': segment_dict, 'lr': 10*cfg.TRAIN_LR}
        ],
		momentum=cfg.TRAIN_MOMENTUM,
		weight_decay=cfg.TRAIN_WEIGHT_DECAY)

	criterion = nn.CrossEntropyLoss(reduce=False, ignore_index=255)
  
	itr = cfg.TRAIN_MINEPOCH * len(dataloader)
	max_itr = cfg.TRAIN_EPOCHS*len(dataloader)
	best_jacc = 0.
	best_epoch = 0
	last_jacc = 0.
	percent_fore = cfg.percent_fore_start/(cfg.TRAIN_EPOCHS-cfg.TRAIN_MINEPOCH)  
	percent_back = cfg.percent_back_start/(cfg.TRAIN_EPOCHS-cfg.TRAIN_MINEPOCH)  
	percent_JA = cfg.percent_JA_start/(cfg.TRAIN_EPOCHS-cfg.TRAIN_MINEPOCH)  
	for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):
		#### start training now
		IoU_array = 0.
		sample_num = 0.
		running_imgwei_loss = 0.0
		running_pixwei_loss = 0.0 
		running_CB_loss = 0.0 
		SPL_loss1 = 0.0
		SPL_loss2 = 0.0
		epoch_samples1 = 0.0
		epoch_samples2 = 0.0
		dataset_list = []
		 
        #########################################################
        ########### give lambda && decay coefficient ############
        #########################################################        
		for i_batch, sample_batched in enumerate(dataloader):
			now_lr = adjust_lr(seg_optimizer, itr, max_itr)
			now_lr = adjust_Wlr(wei_optimizer0, itr, max_itr)
			now_lr = adjust_Wlr(wei_optimizer1, itr, max_itr)
			now_lr = adjust_Wlr(wei_optimizer2, itr, max_itr)
			inputs_batched, labels_batched = sample_batched['image'], sample_batched['segmentation']
			inputs_batched = inputs_batched.cuda()
			labels_batched = labels_batched.long().cuda()
			train_list = {'inputs_batched':inputs_batched, 'labels_batched':labels_batched, 'epoch':epoch, 
					'net':net, 'WeightNet0':WeightNet0, 'WeightNet1':WeightNet1, 'WeightNet2':WeightNet2, 
					'seg_optimizer':seg_optimizer, 'wei_optimizer0':wei_optimizer0, 'wei_optimizer1':wei_optimizer1, 'wei_optimizer2':wei_optimizer2,
					'percent_fore':percent_fore, 'percent_back':percent_back, 'percent_JA':percent_JA}
			
			imgwei_loss, pixwei_loss, CB_loss, CE_fore, CE_back, JA_loss, imgwei, pixwei, result = train_one_batch(train_list)

			IoU, samples = test_one_itr(result, labels_batched)
			IoU_array += IoU
			sample_num += samples
			
			running_imgwei_loss += imgwei_loss.mean().item()
			running_pixwei_loss += pixwei_loss.mean().item()
			running_CB_loss += CB_loss.mean().item()
			samples = torch.sum(torch.abs(pixwei)).cpu().data.numpy()
			samples = samples/result.shape[1]/result.shape[2]  
			SPL_loss1 += (CE_fore+CE_back).item()*samples/2.
			epoch_samples1 +=  samples
			samples = torch.sum(imgwei).cpu().data.numpy()
			SPL_loss2 += JA_loss.item()
			epoch_samples2 +=  samples 
			
			itr += 1

			if (epoch) % 30 == 0:
				name_batched = sample_batched['name']
				[batch, channel, height, width] = inputs_batched.size()
				for i in range(batch):
					inp = inputs_batched[i,:,:,:].cpu().numpy()
					labell = labels_batched[i,:,:].cpu().numpy()
					pred = result[i,:,:]
					v_class1 = pixwei[i,:,:].cpu().data.numpy() * labell
					v_class0 = pixwei[i,:,:].cpu().data.numpy() * (1.-labell)
					dataset_list.append({'v_class1':np.uint8(v_class1*255), 'v_class0':np.uint8(v_class0*255), 'input':np.uint8(inp*255), 
											'name':name_batched[i], 'predict':np.uint8(pred*255), 'label':np.uint8(labell*255)})
		if (epoch) % 30 == 0:
			dataset.save_result_train_weight(dataset_list, cfg.MODEL_NAME)

		IoUP = IoU_array*100/sample_num
		print('Last JA: ', last_jacc, 'This JA: ', IoUP)
		if IoUP > last_jacc:
			percent_fore += cfg.percent_fore_step/(cfg.TRAIN_EPOCHS-cfg.TRAIN_MINEPOCH)  
			percent_back += cfg.percent_back_step/(cfg.TRAIN_EPOCHS-cfg.TRAIN_MINEPOCH)  
			percent_JA += cfg.percent_JA_step/(cfg.TRAIN_EPOCHS-cfg.TRAIN_MINEPOCH)   
	#	else:
	#		percent_fore -= 1/(cfg.TRAIN_EPOCHS-cfg.TRAIN_MINEPOCH)  
	#		percent_back -= 1/(cfg.TRAIN_EPOCHS-cfg.TRAIN_MINEPOCH)  
	#		percent_JA -= 1/(cfg.TRAIN_EPOCHS-cfg.TRAIN_MINEPOCH)   
		last_jacc = IoUP
		percent_fore = np.min([percent_fore, 1.0])
		percent_back = np.min([percent_back, 1.0])
		percent_JA = np.min([percent_JA, 1.0])

		i_batch = i_batch + 1
		#print('epoch:%d/%d\tlambda1:%g\tlambda0:%g\tCE samples:%d\tJA samples:%d\tCE loss:%g\tJA loss:%g\tSPL CE loss:%g\tSPL JA loss:%g' % (epoch, cfg.TRAIN_EPOCHS, 
		#            LambdaFore/i_batch, LambdaBack/i_batch, epoch_samples1, epoch_samples2, running_loss/i_batch, seg_jac_running_loss/i_batch, SPL_loss1/epoch_samples1, SPL_loss2/epoch_samples2))
		print('epoch:%d/%d\tCE samples:%d\tJA samples:%d\tImgWei loss:%g\tPixWei loss:%g\tCB loss:%g\tSPL CE loss:%g\tSPL JA loss:%g' % (epoch, cfg.TRAIN_EPOCHS, 
		            epoch_samples1, epoch_samples2, running_imgwei_loss/i_batch, running_pixwei_loss/i_batch, running_CB_loss/i_batch, SPL_loss1/epoch_samples1, SPL_loss2/epoch_samples2))
		
		#### start testing now
		if (epoch) % 1 == 0:
			IoUP = test_one_epoch(test_dataset, test_dataloader, net, epoch)
			if IoUP > best_jacc:
				model_snapshot_noW(net.state_dict(), new_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_jac%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch,IoUP)),
                	old_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_jac%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,best_epoch,best_jacc)))
			#	model_snapshot(net.state_dict(), WeightNet.state_dict(), new_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_jac%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch,IoUP)),
            #    	old_file=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-%s_%s_%s_epoch%d_jac%.3f.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,best_epoch,best_jacc)),
			#		new_Wfile=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-DWNet-%s_epoch%d_jac%.3f.pth'%(cfg.DATA_NAME,epoch,IoUP)),
			#		old_Wfile=os.path.join(cfg.MODEL_SAVE_DIR,'model-best-DWNet-%s_epoch%d_jac%.3f.pth'%(cfg.DATA_NAME,best_epoch,best_jacc)))
				best_jacc = IoUP
				best_epoch = epoch

def adjust_lr(optimizer, itr, max_itr):
	now_lr = cfg.TRAIN_LR * (1 - (itr/(max_itr+1)) ** cfg.TRAIN_POWER)
#	now_lr = np.max([cfg.TRAIN_LR*0.1, now_lr])
	optimizer.param_groups[0]['lr'] = now_lr
	optimizer.param_groups[1]['lr'] = 30*now_lr
	optimizer.param_groups[2]['lr'] = 10*now_lr
	return now_lr

def adjust_Wlr(optimizer, itr, max_itr):
	now_lr = cfg.TRAIN_LR * (1 - (itr/(max_itr+1)) ** cfg.TRAIN_POWER)
	optimizer.param_groups[0]['lr'] = 10*now_lr
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
			#	predicts = nn.Softmax(dim=1)(predicts.cuda())
				predicts0 = Multi2Binary(predicts0.cuda(), normal_dim=cfg.normal_dim0, polyp_dim=cfg.polyp_dim0)
				predicts1 = Multi2Binary(predicts1.cuda(), normal_dim=cfg.normal_dim1, polyp_dim=cfg.polyp_dim1)
				predicts2 = Multi2Binary(predicts2.cuda(), normal_dim=cfg.normal_dim2, polyp_dim=cfg.polyp_dim2)
				predicts = (predicts0 + predicts1 + predicts2)/3.
				predicts_batched = predicts.clone()
				del predicts0
				del predicts1
			
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