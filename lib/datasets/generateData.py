# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from datasets.ISICDataset import ISICDataset
from datasets.WCEDataset import WCEDataset
from datasets.CVCDataset import CVCDataset
from datasets.CVCEdgeDataset import CVCEdgeDataset

def generate_dataset(dataset_name, cfg, period, aug=False):
	if dataset_name == 'voc2012' or dataset_name == 'VOC2012':
		return VOCDataset('VOC2012', cfg, period, aug)
	elif dataset_name == 'isic2017' or dataset_name == 'ISIC2017':
		return ISICDataset('ISIC2017', cfg, period)
	elif dataset_name == 'wce' or dataset_name == 'WCE':
		return WCEDataset('WCE', cfg, period)
	elif dataset_name == 'cvc' or dataset_name == 'CVC':
		return CVCDataset('CVC', cfg, period)
	elif dataset_name == 'cvcedge' or dataset_name == 'CVCEdge':
		return CVCEdgeDataset('CVC', cfg, period)
	else:
		raise ValueError('generateData.py: dataset %s is not support yet'%dataset_name)
