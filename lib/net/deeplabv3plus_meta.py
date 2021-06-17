# ----------------------------------------
# Written by Xiaoqing Guo
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from net.sync_batchnorm import SynchronizedBatchNorm2d
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn import init

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)
    #    ignore = SynchronizedBatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
        
class Bottleneck(MetaModule):
    expansion = 4
    bn_mom = 0.0003

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes, momentum=self.bn_mom)

        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1*atrous, dilation=atrous, bias=False)
        self.bn2 = MetaBatchNorm2d(planes, momentum=self.bn_mom)

        self.conv3 = MetaConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes * self.expansion, momentum=self.bn_mom)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Atrous(MetaModule):

    def __init__(self, block, layers, atrous=None, os=16, bn_mom=0.1):
        super(ResNet_Atrous, self).__init__()
        stride_list = None
        if os == 8:
            stride_list = [2,1,1]
        elif os == 16:
            stride_list = [2,2,1]
        else:
            raise ValueError('resnet_atrous.py: output stride=%d is not supported.'%os) 
            
        self.inplanes = 64
        self.bn_mom = bn_mom
        self.conv1 = MetaConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = MetaBatchNorm2d(64, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=stride_list[0])
        self.layer3 = self._make_layer(block, 512, 256, layers[2], stride=stride_list[1], atrous=16//os)
        self.layer4 = self._make_layer(block, 1024, 512, layers[3], stride=stride_list[2], atrous=[item*16//os for item in atrous])
        self.layers = []

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, MetaBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self):
        return self.layers

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, atrous=None):
        downsample = None
        if atrous == None:
            atrous = [1]*blocks
        elif isinstance(atrous, int):
            atrous_list = [atrous]*blocks
            atrous = atrous_list
        if stride != 1 or inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                MetaConv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(planes * block.expansion, momentum=self.bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride=stride, atrous=atrous[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes*block.expansion, planes, stride=1, atrous=atrous[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        self.layers.append(x)
        x = self.layer2(x)
        self.layers.append(x)
        x = self.layer3(x)
        self.layers.append(x)
        x = self.layer4(x)
        self.layers.append(x)

        return x


class ASPP(MetaModule):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				MetaConv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				MetaBatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				MetaConv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate,bias=True),
				MetaBatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				MetaConv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate,bias=True),
				MetaBatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				MetaConv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate,bias=True),
				MetaBatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = MetaConv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = MetaBatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)
		self.conv_cat = nn.Sequential(
				MetaConv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				MetaBatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b,c,row,col] = x.size()
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row,col), None, 'bilinear', True)
		
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class deeplabv3plus_meta(MetaModule):
	def __init__(self, cfg):
		super(deeplabv3plus_meta, self).__init__()
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
				MetaConv2d(indim, cfg.MODEL_SHORTCUT_DIM, kernel_size=cfg.MODEL_SHORTCUT_KERNEL, stride=1, padding=cfg.MODEL_SHORTCUT_KERNEL//2,bias=True),
				MetaBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),		
		)		
		self.cat_conv = nn.Sequential(
				MetaConv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, kernel_size=3, stride=1, padding=1,bias=True),
				MetaBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				MetaConv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, kernel_size=3, stride=1, padding=1,bias=True),
				MetaBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
#		self.cls_conv = MetaConv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		self.cls_conv_0 = MetaConv2d(cfg.MODEL_ASPP_OUTDIM, cfg.normal_dim0 + cfg.polyp_dim0, 1, 1, padding=0, bias=False)
		self.cls_conv_1 = MetaConv2d(cfg.MODEL_ASPP_OUTDIM, cfg.normal_dim1 + cfg.polyp_dim1, 1, 1, padding=0, bias=False)
		self.cls_conv_2 = MetaConv2d(cfg.MODEL_ASPP_OUTDIM, cfg.normal_dim2 + cfg.polyp_dim2, 1, 1, padding=0, bias=False)
		for m in self.modules():
			if isinstance(m, MetaConv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, MetaBatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		self.backbone = ResNet_Atrous(Bottleneck, [3, 4, 23, 3], atrous=[2,2,2], os=cfg.MODEL_OUTPUT_STRIDE, bn_mom = cfg.TRAIN_BN_MOM)
		self.backbone_layers = self.backbone.get_layers()

	def forward(self, x):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)
		feature_aspp = self.upsample_sub(feature_aspp)

		feature_shallow = self.shortcut_conv(layers[0])
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		feature = self.cat_conv(feature_cat) 
		result = self.cls_conv_0(feature)
		result0 = self.upsample4(result)
		result = self.cls_conv_1(feature)
		result1 = self.upsample4(result)
		result = self.cls_conv_2(feature)
		result2 = self.upsample4(result)
		return feature, result0, result1, result2

class LearnSPL_meta(MetaModule):
    def __init__(self, cfg):
        super(LearnSPL_meta, self).__init__()

        self.cfg = cfg
        self.label_emb = nn.Embedding(cfg.MODEL_NUM_CLASSES, 2)
        self.JA_percent_emb = nn.Embedding(101, 4)
        self.CE_percent_fore_emb = nn.Embedding(101, 5)
        self.CE_percent_back_emb = nn.Embedding(101, 5)

        self.lstm = nn.LSTM(2, 10, 1, bidirectional=True)

        self.fc1 = MetaLinear(1, 20)
        self.fc2 = MetaLinear(20, 1)

        self.weight_conv = nn.Sequential(
				MetaConv2d(258, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				MetaBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				MetaConv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				MetaBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
                nn.Tanh(),
		)
        self.weight_cls_conv = nn.Sequential(
                MetaConv2d(cfg.MODEL_ASPP_OUTDIM, 1, 1, 1, padding=0),
                nn.Sigmoid(),
        )
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, MetaBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, JA_loss, CE_loss, feature_maps, labels):        
        ### image-level weight
        JA_loss = torch.unsqueeze(JA_loss, 1).cuda()
        x = JA_loss

        img = F.tanh(self.fc1(x))
        img_weight = F.sigmoid(self.fc2(img))

        ### pixel-level weight
        imgresize = torch.nn.UpsamplingBilinear2d(size=(int(self.cfg.DATA_RESCALE/4),int(self.cfg.DATA_RESCALE/4)))
        labels = imgresize(torch.unsqueeze(labels, 1).float())
        CE_loss = imgresize(torch.unsqueeze(CE_loss, 1))
        feature_cat = torch.cat([feature_maps,labels,CE_loss],1)

        result = self.weight_conv(feature_cat) 
        result = self.weight_cls_conv(result)
        pix_weight = self.upsample4(result)

        return img_weight, pix_weight