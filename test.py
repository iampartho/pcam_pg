import sys
import os
import argparse
import logging
import json
import time
import subprocess
from shutil import copyfile
import cv2
import numpy as np
from sklearn import metrics
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import DataParallel
from skimage.measure import label
from tensorboardX import SummaryWriter
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../') # then file is in bin so it sets the system path to chesXpert

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from data.dataset import ImageDataset  # noqa
from data.utils import transform
from model.classifier_agcnn1 import Classifier  # noqa
from utils.misc import lr_schedule  # noqa
from model.utils import get_optimizer  # noqa
#from model.classifier_agcnn import *
from model.classifier_agcnn_fusion import *
import torchvision.transforms as transforms
from model.utils import tensor2numpy
import pandas as pd

def create_img(path, cfg):
	image = cv2.imread(path, 0) # reading images in grayscale
    image = Image.fromarray(image) # converting it to pil image
    # if self._mode == 'train': # No TTA used
    #     image = GetTransforms(image, type=self.cfg.use_transforms_type)
    image = np.array(image)
    image = transform(image, cfg) # this is happening for test time as well
    

    return image

def Attention_gen_patchs(ori_image, fm_cuda):
    # feature map -> feature mask (using feature map to crop on the original image) -> crop -> patchs
    # fm_cuda => apatoto classifier er feat_map pass kortese
    # feature_conv = fm_cuda.data.cpu().numpy()
    # size_upsample = (256, 256) 
    # bz, nc, h, w = feature_conv.shape

    # patchs_cuda = torch.FloatTensor().cuda()

    # for i in range(0, bz):
    #     feature = feature_conv[i]
    #     cam = feature.reshape((nc, h*w))
    #     cam = cam.sum(axis=0)
    #     cam = cam.reshape(h,w)
    #     cam = cam - np.min(cam)
    #     cam_img = cam / np.max(cam)
    #     cam_img = np.uint8(255 * cam_img)

    #     heatmap_bin = binImage(cv2.resize(cam_img, size_upsample))
    #     heatmap_maxconn = selectMaxConnect(heatmap_bin)
    #     heatmap_mask = heatmap_bin * heatmap_maxconn
    #     #print(heatmap_mask)
        
    #     ind = np.argwhere(heatmap_mask != 0)
    #     if not ind==[] :
    #       minh = 0
    #       minw = 0
    #       maxh = size_upsample[0]
    #       maxw = size_upsample[1]
    #     else :
    #       minh = min(ind[:,0])
    #       minw = min(ind[:,1])
    #       maxh = max(ind[:,0])
    #       maxw = max(ind[:,1])
        
    #     # to ori image 
    #     #print('xxxxxxxxxxxxxxxx')
    #     # print(ori_image[i].shape)
    #     # ori_img = ori_image[i].permute(1,2,0)
    #     # print(ori_image[i].shape)
    #     image = ori_image[i].numpy().reshape(256,256,3)
    #     image = image[int(256*0.334):int(256*0.667),int(256*0.334):int(256*0.667),:]

    #     image = cv2.resize(image, size_upsample)
    #     image_crop = image[minh:maxh,minw:maxw,:] * 256 # because image was normalized before
    #     image_crop = preprocess(Image.fromarray(image_crop.astype('uint8')).convert('RGB')) 

    #     img_variable = torch.autograd.Variable(image_crop.reshape(3,256,256).unsqueeze(0).cuda())

    #     patchs_cuda = torch.cat((patchs_cuda,img_variable),0)

    fm_cuda = torch.stack(fm_cuda)
    feature_conv = tensor2numpy(torch.sigmoid(fm_cuda))
    size_upsample = (256, 256) 
    num_cls, bz, h, w = feature_conv.shape

    patchs_cuda = torch.FloatTensor().cuda()

    for i in range(0, bz):
        all_idx=np.array([])
        for j in range(num_cls):
            feature = feature_conv[j, i, :, :]
            # cam = feature.reshape((nc, h*w))
            # cam = cam.sum(axis=0)
            cam = feature.reshape(h,w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)

            heatmap_bin = binImage(cv2.resize(cam_img, size_upsample))
            heatmap_maxconn = selectMaxConnect(heatmap_bin)
            heatmap_mask = heatmap_bin * heatmap_maxconn
            #print(heatmap_mask)
            
            ind = np.argwhere(heatmap_mask != 0)
            if j==0:
                all_idx = ind
            else:

                np.concatenate((all_idx, ind), axis=0)
                #all_idx += ind

        if len(all_idx)==0 :
          minh = 0
          minw = 0
          maxh = size_upsample[0]
          maxw = size_upsample[1]
        else :
          minh = min(all_idx[:,0])
          minw = min(all_idx[:,1])
          maxh = max(all_idx[:,0])
          maxw = max(all_idx[:,1])
        
        # to ori image 
        #print('xxxxxxxxxxxxxxxx')
        # print(ori_image[i].shape)
        # ori_img = ori_image[i].permute(1,2,0)
        # print(ori_image[i].shape)
        image = ori_image[i].numpy().reshape(256,256,3)
        #image = image[int(256*0.334):int(256*0.667),int(256*0.334):int(256*0.667),:]

        image = cv2.resize(image, size_upsample)
        image_crop = image[minh:maxh,minw:maxw,:] * 256 # because image was normalized before
        image_crop = preprocess(Image.fromarray(image_crop.astype('uint8')).convert('RGB')) 

        img_variable = torch.autograd.Variable(image_crop.reshape(3,256,256).unsqueeze(0).cuda())

        patchs_cuda = torch.cat((patchs_cuda,img_variable),0)

    return patchs_cuda

def binImage(heatmap):
    _, heatmap_bin = cv2.threshold(heatmap , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # t in the paper
    #_, heatmap_bin = cv2.threshold(heatmap , 178 , 255 , cv2.THRESH_BINARY)
    return heatmap_bin


def selectMaxConnect(heatmap):
    labeled_img, num = label(heatmap, connectivity=2, background=0, return_num=True)    
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    if max_num == 0:
       lcc = (labeled_img == -1)
    lcc = lcc + 0
    return lcc 


def test(model_global, model_local, model_fusion, csv_path, cfg):
	df = pd.read_csv(csv_path)
	all_img_path = df['Path'].values

	pred_array = np.zeros((len(all_img_path),5), dtype=np.float64)

	for img_idx, each_path in enumerate(all_img_path):
		image = create_img(each_path, cfg)

		image = image.to(device)

		with torch.no_grad():
            output_global,feat_list_global, feat_map, logit_maps = model_global(image)
            #print(image_v.shape)
            patch_var = Attention_gen_patchs(image_v,logit_maps)
            output_local,feat_list_local,feat_map_local,_ = model_local(patch_var)
        
            output = model_fusion(feat_map, feat_map_local)

        
        for i in range(5):
            prob = torch.sigmoid(output[i])
            prob = tensor2numpy(prob)

            pred_array[img_idx, i] = prob

        return pred_array



def main(global_weight_path, local_weight_path, fusion_weight_path, csv_path):
    with open('./config/example_PCAM.json') as f:
        cfg = edict(json.load(f))

    device = torch.device('cuda:0')

    model_global = Classifier(cfg)
    model_local = Classifier(cfg)
    model_fusion = Classifier_F(cfg)

    model_global = DataParallel(model_global, device_ids=device_ids).to(device).eval()
    model_local = DataParallel(model_local, device_ids=device_ids).to(device).eval()
    model_fusion = DataParallel(model_fusion, device_ids=device_ids).to(device).eval()


    ckpt = torch.load(global_weight_path)
    model_global.module.load_state_dict(ckpt['state_dict'])

    ckpt = torch.load(local_weight_path, map_location=device)
    model_local.module.load_state_dict(ckpt['state_dict'])

    ckpt = torch.load(fusion_weight_path, map_location=device)
    model_fusion.module.load_state_dict(ckpt['state_dict'])

    predicted_array = test(model_global, model_local, model_fusion, csv_path, cfg)

if __name__ == '__main__':
	global_weight_path = ''
	local_weight_path = ''
	fusion_weight_path = ''
	csv_path = ''
    main(global_weight_path, local_weight_path, fusion_weight_path, csv_path)