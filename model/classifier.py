from torch import nn

import torch.nn.functional as F
from model.backbone.vgg import (vgg19, vgg19_bn)
from model.backbone.densenet import (densenet121, densenet169, densenet201)
from model.backbone.inception import (inception_v3)
from model.global_pool import GlobalPool
from model.attention_map import AttentionMap
import torch

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


class Classifier(nn.Module):

    def __init__(self, cfg):
        super(Classifier, self).__init__()
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
        self._init_classifier()
        self._init_bn() #initializing the batch-normalization
        self._init_attention_map()

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

    def forward(self, x):
        # (N, C, H, W)
        feat_map = self.backbone(x) # according to the i/p size it returns [N, 1024, H, W]
                                    # for 224x224 it returns 7x7 and for 256x256 it returns 8x8 and for 512x512 it returns 16x16
        
        if self.cfg.attention_map != "None":
            feat_map = self.attention_map(feat_map)
        # [(N, 1), (N,1),...]
        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = list()
        for index, num_class in enumerate(self.cfg.num_classes):
             # this seems problematic

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

        return (logits, logit_maps)
