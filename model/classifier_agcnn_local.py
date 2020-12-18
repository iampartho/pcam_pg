from torch import nn

import torch.nn.functional as F
from model.backbone.vgg import (vgg19, vgg19_bn)
from model.backbone.densenet import (densenet121, densenet169, densenet201)
from model.backbone.inception import (inception_v3)
from model.global_pool import GlobalPool
from model.attention_map import AttentionMap
import torch

############ Jawad & Iqbal's Code starts here ##############

import torch
import torch.nn as nn
import math
from torch.nn.functional import conv2d




#The kernel_size, var_x, var_y all should be torch_tensors
def gaussian_kernel(kernel_size, var_x, var_y):
    ax = torch.round(torch.linspace(-math.floor(kernel_size/2), math.floor(kernel_size/2),
                                    kernel_size), out=torch.FloatTensor())
    x = ax.view(1, -1).repeat(ax.size(0), 1).to('cuda')
    y = ax.view(-1, 1).repeat(1, ax.size(0)).to('cuda')
    x2 = torch.pow(x,2)
    y2 = torch.pow(y,2)
    std_x = torch.pow(var_x, 0.5)
    std_y = torch.pow(var_y, 0.5)
    temp = - ((x2/var_x) + (y2/var_y)) / 2
    kernel = torch.exp(temp)/(2*math.pi*std_x*std_y)
    kernel = kernel/kernel.sum()
    return kernel

# 1, 1
    
class GB2d(nn.Module):
    def __init__(self, in_channel, kernel_size, var_x = None, var_y = None, aug_train = True, padding = 0):
        super(GB2d, self).__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.aug_train = aug_train
        
        if self.aug_train:
            self.var_x = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
            self.var_y = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
            self.init_parameters()
        else:
            self.var_x = torch.FloatTensor([var_x]).to('cuda')
            self.var_y = torch.FloatTensor([var_y]).to('cuda')
            self.kernel = gaussian_kernel(self.kernel_size, self.var_x, self.var_y)
            self.kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
        
            
    def init_parameters(self):
        self.var_x.data.uniform_(0, 1)
        self.var_y.data.uniform_(0, 1)
        
    def forward(self, input):
        input_shape = input.shape
        batch_size, h, w = input_shape[0], input_shape[2], input_shape[3]
        
        if self.aug_train:
            self.kernel = gaussian_kernel(self.kernel_size, self.var_x, self.var_y)
            self.kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
            
        output = conv2d(input.view(batch_size*self.in_channel,1,h,w), self.kernel, padding = self.padding)
        h_out = math.floor((h - self.kernel_size + 2*self.padding)+1)
        w_out = math.floor((w - self.kernel_size + 2*self.padding)+1)
        output = output.view(batch_size, self.in_channel, h_out, w_out)
        return output


class USM2d(nn.Module):
    def __init__(self, in_channel, kernel_size, var_x = None, var_y = None, alpha = None, aug_train = True):
        super(USM2d, self).__init__()
        self.padding = int((kernel_size - 1) / 2)
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.aug_train = aug_train
        
        if self.aug_train:
            self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
            self.var_x = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
            self.var_y = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
            self.init_parameters()
        else:
            self.alpha = torch.FloatTensor([alpha])
            self.var_x = torch.FloatTensor([var_x])
            self.var_y = torch.FloatTensor([var_y])
            self.GB_fcn = GB2d(in_channel, kernel_size, var_x, 
                               var_y, train = False, padding = self.padding)
    
    def init_parameters(self):
        self.alpha.data.uniform_(0,1)
        self.var_x.data.uniform_(0,1)
        self.var_y.data.uniform_(0,1)
        
    def forward(self, input_img):      
        if self.aug_train:
            self.GB_fcn = GB2d(self.in_channel, self.kernel_size, self.var_x, 
                               self.var_y, aug_train = False, padding = self.padding)
            blur_img = self.GB_fcn(input_img)
            usm_img = (1 + torch.sigmoid(self.alpha)) * input_img - torch.sigmoid(self.alpha) * blur_img
        else:
            blur_img = self.GB_fcn(input_img)
            usm_img = (1 + torch.sigmoid(self.alpha)) * input_img - torch.sigmoid(self.alpha) * blur_img
            
        return usm_img

################# And ends here ######################################


BACKBONES = {'vgg19': vgg19,
             'vgg19_bn': vgg19_bn,
             'densenet121': densenet121,
             'densenet169': densenet169,
             'densenet201': densenet201,
             'inception_v3': inception_v3}


BACKBONES_TYPES = {'vgg19': 'vgg',
                   'vgg19_bn': 'vgg',
                   'densenet121': 'densenet',
                   'densenet169': 'densenet',
                   'densenet201': 'densenet',
                   'inception_v3': 'inception'}


class Classifier_local(nn.Module):

    def __init__(self, cfg):
        super(Classifier_local, self).__init__()
        self.cfg = cfg
        self.backbone = BACKBONES[cfg.backbone](cfg)
        #trained_kernel = self.backbone.features.conv0.weight #densenet specific
        #new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # changing the conv0
        #with torch.no_grad():
        #    new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*1, dim=1) # loading the weights
        #self.backbone.features.conv0 = new_conv
        self.global_pool = GlobalPool(cfg)
        self.expand = 1
        if cfg.global_pool == 'AVG_MAX':
            self.expand = 2
        elif cfg.global_pool == 'AVG_MAX_LSE':
            self.expand = 3 #the expand variable depends upon number of pulling used in the code
        self.normalize = nn.BatchNorm2d(self.backbone.num_features * self.expand)
        self._init_classifier()
        self._init_bn() #initializing the batch-normalization
        self._init_attention_map()
        self.aug_layer = [USM2d(3,3) ,
                        GB2d(3, 3, aug_train = True, padding = 1)]



    def _init_classifier(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        512 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True)) # this is the 1x1 convolution refered in the paper
                                    # basically they have used 5 , 1x1 conv to comprehend each class
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                setattr(
                    self,
                    "fc_" +
                    str(index),
                    nn.Conv2d(
                        self.backbone.num_features *
                        self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        2048 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.cfg.backbone)
                )

            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01) #mean and std_deviation
                                                        # but if we change the FC how will it change correspondingly
                classifier.bias.data.zero_()

    def _init_bn(self): 
        for index, num_class in enumerate(self.cfg.num_classes):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                setattr(self, "bn_" + str(index),
                        nn.BatchNorm2d(512 * self.expand))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                setattr(
                    self,
                    "bn_" +
                    str(index),
                    nn.BatchNorm2d(
                        self.backbone.num_features *
                        self.expand))
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                setattr(self, "bn_" + str(index),
                        nn.BatchNorm2d(2048 * self.expand))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.cfg.backbone)
                )

    def _init_attention_map(self):
        if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
            setattr(self, "attention_map", AttentionMap(self.cfg, 512))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
            setattr(
                self,
                "attention_map",
                AttentionMap(
                    self.cfg,
                    self.backbone.num_features))
        elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
            setattr(self, "attention_map", AttentionMap(self.cfg, 2048))
        else:
            raise Exception(
                'Unknown backbone type : {}'.format(self.cfg.backbone)
            )

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, x, y):
        
        y=self.normalize(y)

        # [(N, 1), (N,1),...] 
        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = list()
        feat_list = list()
        for index, num_class in enumerate(self.cfg.num_classes):
             # this seems problematic

            if index in [0, 2, 4]
                x[index] = self.aug_layer[0]
            else:
                x[index] = self.aug_layer[1]
            # (N, C, H, W)
            feat_map = self.backbone(x[index]) # according to the i/p size it returns [N, 1024, H, W]
                                    # for 224x224 it returns 7x7 and for 256x256 it returns 8x8 and for 512x512 it returns 16x16
            feat_map = self.normalize(feat_map)
            feat_map = feat_map + y
            if self.cfg.attention_map != "None":
                feat_map = self.attention_map(feat_map)

            classifier = getattr(self, "fc_" + str(index))
            #attention_weight = classifier.weight.data
            #attention_weight = classifier.weight.data.normal_(0, 0.01)

            #attentioned_feat_map = torch.mul(feat_map, attention_weight)
            # (N, 1, H, W)
            logit_map = None
            if not (self.cfg.global_pool == 'AVG_MAX' or
                    self.cfg.global_pool == 'AVG_MAX_LSE'):
                logit_map = classifier(feat_map)
                logit_maps.append(logit_map.squeeze())

            # f

            # (N, C, 1, 1)
            feat = self.global_pool(feat_map, logit_map)

            if self.cfg.fc_bn:
                bn = getattr(self, "bn_" + str(index))
                feat = bn(feat)
            feat = F.dropout(feat, p=self.cfg.fc_drop, training=self.training)
            # (N, num_class, 1, 1)

            logit = classifier(feat)
            # (N, num_class)
            logit = logit.squeeze(-1).squeeze(-1)
            logits.append(logit)
            feat_list.append(feat)

        return (logits, feat_list, feat_map, logit_maps)
