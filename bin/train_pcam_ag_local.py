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
from model.classifier_agcnn1 import Classifier
from model.classifier_agcnn_local import Classifier_local  # noqa
from utils.misc import lr_schedule  # noqa
from model.utils import get_optimizer  # noqa
#from model.classifier_agcnn import *
#from model.classifier_agcnn_fusion import *
import torchvision.transforms as transforms
from model.utils import tensor2numpy

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format") # args.cfg_path
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help="Path to the saved models")
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0,1,2,3', type=str,
                    help="GPU indices ""comma separated, e.g. '0,1' ")
parser.add_argument('--pre_train_gloabl', default="/content/pcam_pg/best1.ckpt", type=str, help="If get"
                    "parameters from pretrained model")
parser.add_argument('--pre_train_local', default="/content/drive/MyDrive/learning_chexpert/best_local1.ckpt", type=str, help="If get"
                    "parameters from pretrained model")
parser.add_argument('--resume', default=0, type=int, help="If resume from "
                    "previous run")
parser.add_argument('--logtofile', default=False, type=bool, help="Save log "
                    "in save_path/log.txt if set True")
parser.add_argument('--verbose', default=False, type=bool, help="Detail info")

def get_loss(output, target, index, device, cfg): # index refer to the nth class number
    if cfg.criterion == 'BCE':
        for num_class in cfg.num_classes:
            assert num_class == 1
        target = target[:, index].view(-1)
        pos_weight = torch.from_numpy(
            np.array(cfg.pos_weight,
                     dtype=np.float32)).to(device).type_as(target)
        if cfg.batch_weight:
            if target.sum() == 0:
                loss = torch.tensor(0., requires_grad=True).to(device)
            else:
                weight = (target.size()[0] - target.sum()) / target.sum()
                loss = F.binary_cross_entropy_with_logits(
                    output[index].view(-1), target, pos_weight=weight)
        else:
            loss = F.binary_cross_entropy_with_logits(
                output[index].view(-1), target, pos_weight=pos_weight[index]) 

        label = torch.sigmoid(output[index].view(-1)).ge(0.5).float() # this returning the output labels for 'index' class after thresholding
        acc = (target == label).float().sum() / len(label)
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))

    return (loss, acc)
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(256),
   transforms.ToTensor(),
   normalize,
])
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

    #patchs_cuda = torch.FloatTensor().cuda()

    all_patches = []

    
        #all_idx=np.array([])
    for j in range(num_cls):
        patchs_cuda = torch.FloatTensor().cuda()
        for i in range(0, bz):
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
            # if j==0:
            #     all_idx = ind
            # else:

            #     np.concatenate((all_idx, ind), axis=0)
                #all_idx += ind

            if len(ind)==0 :
              minh = 0
              minw = 0
              maxh = size_upsample[0]
              maxw = size_upsample[1]
            else :
              minh = min(ind[:,0])
              minw = min(ind[:,1])
              maxh = max(ind[:,0])
              maxw = max(ind[:,1])
            
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

        all_patches.append(patchs_cuda)
    #return patchs_cuda
    return all_patches
    

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

def train_epoch(summary, summary_dev, cfg, args, model_global,model_local, dataloader,
                dataloader_dev, optimizer_global,optimizer_local, summary_writer, best_dict,
                dev_header):
    torch.set_grad_enabled(True)
    #model_global.train()
    model_local.train()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    label_header = dataloader.dataset._label_header
    num_tasks = len(cfg.num_classes) # this is 5

    time_now = time.time()
    loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)
    for step in range(steps):
        image_v, target = next(dataiter)
        image = image_v.to(device)
        target = target.to(device)
        with torch.no_grad():
            output_global,feat_list_global, feat_map, logit_maps = model_global(image)
        #print(image_v.shape)
        patch_var = Attention_gen_patchs(image_v,logit_maps)
        output_local,_,_, _ = model_local(patch_var, feat_map)
        


        # different number of tasks
        loss_global = 0
        loss_local = 0
        loss_fusion = 0

        
        for t in range(num_tasks):
            loss_t, acc_t = get_loss(output_local, target, t, device, cfg) # loss_t and acc_t is for class 't'
            loss_local += loss_t
            loss_sum[t] += loss_t.item()
            acc_sum[t] += acc_t.item()

        


        loss =loss_local
        optimizer_local.zero_grad()
        

        loss.backward()

        optimizer_local.step()
        

        summary['step'] += 1

        if summary['step'] % cfg.log_every == 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            loss_sum /= cfg.log_every
            acc_sum /= cfg.log_every
            loss_str = ' '.join(map(lambda x: '{:.5f}'.format(x), loss_sum))
            acc_str = ' '.join(map(lambda x: '{:.3f}'.format(x), acc_sum))

            logging.info(
                '{}, Train, Epoch : {}, Step : {}, Loss : {}, '
                'Acc : {}, Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['epoch'] + 1, summary['step'], loss_str,
                        acc_str, time_spent))
            print(
                '{}, Train, Epoch : {}, Step : {}, Loss : {}, '
                'Acc : {}, Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['epoch'] + 1, summary['step'], loss_str,
                        acc_str, time_spent))

            for t in range(num_tasks):
                summary_writer.add_scalar(
                    'train/loss_{}'.format(label_header[t]), loss_sum[t],
                    summary['step'])
                summary_writer.add_scalar(
                    'train/acc_{}'.format(label_header[t]), acc_sum[t],
                    summary['step'])

            loss_sum = np.zeros(num_tasks) # after logging the variable are re-initialize so 
            acc_sum = np.zeros(num_tasks) # the logging information is for that batch span only

        if summary['step'] % cfg.test_every == 0:
            time_now = time.time()
            summary_dev, predlist, true_list = test_epoch(
                summary_dev, cfg, args,model_global, model_local, dataloader_dev)
            # pred_list has the probabilities(non-binarized) of each classes in their index for all val images
            # true_list has the g_t of each classes in their index binarized
            time_spent = time.time() - time_now # apatoto eituku dekhsi

            auclist = []
            for i in range(len(cfg.num_classes)):
                y_pred = predlist[i]
                y_true = true_list[i]
                fpr, tpr, thresholds = metrics.roc_curve(
                    y_true, y_pred, pos_label=1) # FP rate, TP rate, Thresholds
                auc = metrics.auc(fpr, tpr) # auc per class
                auclist.append(auc)
            summary_dev['auc'] = np.array(auclist)

            loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                        summary_dev['loss']))
            acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['acc']))
            auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['auc']))

            logging.info(
                '{}, Dev, Step : {}, Loss : {}, Acc : {}, Auc : {},'
                'Mean auc: {:.3f} ''Run Time : {:.2f} sec' .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary['step'],
                    loss_dev_str,
                    acc_dev_str,
                    auc_dev_str,
                    summary_dev['auc'].mean(),
                    time_spent))
            print(
                '{}, Dev, Step : {}, Loss : {}, Acc : {}, Auc : {},'
                'Mean auc: {:.3f} ''Run Time : {:.2f} sec' .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary['step'],
                    loss_dev_str,
                    acc_dev_str,
                    auc_dev_str,
                    summary_dev['auc'].mean(),
                    time_spent))

            for t in range(len(cfg.num_classes)):
                summary_writer.add_scalar(
                    'dev/loss_{}'.format(dev_header[t]),
                    summary_dev['loss'][t], summary['step'])
                summary_writer.add_scalar(
                    'dev/acc_{}'.format(dev_header[t]), summary_dev['acc'][t],
                    summary['step'])
                summary_writer.add_scalar(
                    'dev/auc_{}'.format(dev_header[t]), summary_dev['auc'][t],
                    summary['step'])

            save_best = False
            mean_acc = summary_dev['acc'][cfg.save_index].mean() # cfg.save_index = [0,1,2,3,4]
            if mean_acc >= best_dict['acc_dev_best']:
                best_dict['acc_dev_best'] = mean_acc
                if cfg.best_target == 'acc':
                    save_best = True

            mean_auc = summary_dev['auc'][cfg.save_index].mean()
            if mean_auc >= best_dict['auc_dev_best']:
                best_dict['auc_dev_best'] = mean_auc
                if cfg.best_target == 'auc': #cfg.best_target = 'auc'
                    save_best = True

            mean_loss = summary_dev['loss'][cfg.save_index].mean()
            if mean_loss <= best_dict['loss_dev_best']:
                best_dict['loss_dev_best'] = mean_loss
                if cfg.best_target == 'loss':
                    save_best = True

            if save_best:
                torch.save(
                    {'epoch': summary['epoch'],
                     'step': summary['step'],
                     'acc_dev_best': best_dict['acc_dev_best'],
                     'auc_dev_best': best_dict['auc_dev_best'],
                     'loss_dev_best': best_dict['loss_dev_best'],
                     'state_dict': model_local.module.state_dict()},
                    os.path.join("/content/drive/MyDrive/learning_chexpert", 'best_local{}.ckpt'.format(
                        best_dict['best_idx']))
                )
                best_dict['best_idx'] += 1
                if best_dict['best_idx'] > cfg.save_top_k:
                    best_dict['best_idx'] = 1
                logging.info(
                    '{}, Best, Step : {}, Loss : {}, Acc : {},Auc :{},'
                    'Best Auc : {:.3f}' .format(
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['step'],
                        loss_dev_str,
                        acc_dev_str,
                        auc_dev_str,
                        best_dict['auc_dev_best']))
                print(
                    '{}, Best, Step : {}, Loss : {}, Acc : {},Auc :{},'
                    'Best Auc : {:.3f}' .format(
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['step'],
                        loss_dev_str,
                        acc_dev_str,
                        auc_dev_str,
                        best_dict['auc_dev_best']
                       ))
        #model_global.train()
        model_local.train()
        
        torch.set_grad_enabled(True)
    summary['epoch'] += 1

    return summary


def test_epoch(summary, cfg, args, model_global,model_local, dataloader):
    torch.set_grad_enabled(False)
    model_global.eval()
    model_local.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)

    loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)

    predlist = list(x for x in range(len(cfg.num_classes)))
    true_list = list(x for x in range(len(cfg.num_classes)))
    for step in range(steps):
        image_v, target = next(dataiter)
        image = image_v.to(device)
        target = target.to(device)
        output_global,feat_list_global, feat_map, logit_maps = model_global(image)
        #print(image_v.shape)
        patch_var = Attention_gen_patchs(image_v,logit_maps)
        output_local,feat_list_local,_,_ = model_local(patch_var, feat_map)
        
        # different number of tasks
        for t in range(len(cfg.num_classes)):

            loss_t, acc_t = get_loss(output_local, target, t, device, cfg)
            # AUC
            output_tensor = torch.sigmoid(
                output_local[t].view(-1)).cpu().detach().numpy()
            target_tensor = target[:, t].view(-1).cpu().detach().numpy()
            if step == 0:
                predlist[t] = output_tensor
                true_list[t] = target_tensor
            else:
                predlist[t] = np.append(predlist[t], output_tensor)
                true_list[t] = np.append(true_list[t], target_tensor)

            loss_sum[t] += loss_t.item()
            acc_sum[t] += acc_t.item()
    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps

    return summary, predlist, true_list


def run(args):
    with open(args.cfg_path) as f:
        cfg = edict(json.load(f)) # now by using cfg.JSON_KEY we can access all the values stored in the key
        if args.verbose is True:
            print(json.dumps(cfg, indent=4)) #printing all the configuration

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path) # creating folders to save the models
    if args.logtofile is True:
        logging.basicConfig(filename=args.save_path + '/log.txt',
                            filemode="w", level=logging.INFO) # creating a logging to log the training information
    else:
        logging.basicConfig(level=logging.INFO)

    if not args.resume:
        with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:
            json.dump(cfg, f, indent=1)

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
            .format(num_devices, len(device_ids)))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    model_global = Classifier(cfg)
    model_local = Classifier_local(cfg)
    #model_fusion = Classifier_F(cfg) # model is done
    if args.verbose is True:
        from torchsummary import summary
        if cfg.fix_ratio:
            h, w = cfg.long_side, cfg.long_side
        else:
            h, w = cfg.height, cfg.width
        summary(model_global.to(device), (3, h, w)) # showing he full model summary
    model_global = DataParallel(model_global, device_ids=device_ids).to(device).train()
    model_local = DataParallel(model_local, device_ids=device_ids).to(device).train()
    model_global.eval()
    #model_fusion = DataParallel(model_fusion, device_ids=device_ids).to(device).train()
    if args.pre_train_gloabl is not None:
        if os.path.exists(args.pre_train_gloabl):
            ckpt = torch.load(args.pre_train_gloabl, map_location=device)
            model_global.module.load_state_dict(ckpt['state_dict'])

    if args.pre_train_local is not None:
        if os.path.exists(args.pre_train_gloabl):

            ckpt = torch.load(args.pre_train_local, map_location=device)
            model_local.module.load_state_dict(ckpt['state_dict'])
            print('pretrained local  loaded')

    optimizer_global = get_optimizer(model_global.parameters(), cfg)
    optimizer_local = get_optimizer(model_local.parameters(), cfg)
    #optimizer_fusion = get_optimizer(model_fusion.parameters(), cfg)


    # ei 10 line Ana bujhabe
    # src_folder = os.path.dirname(os.path.abspath(__file__)) + '/../'
    # dst_folder = os.path.join(args.save_path, 'classification')
    # rc, size = subprocess.getstatusoutput('du --max-depth=0 %s | cut -f1'
    #                                       % src_folder)
    # if rc != 0:
    #     raise Exception('Copy folder error : {}'.format(rc))
    # rc, err_msg = subprocess.getstatusoutput('cp -R %s %s' % (src_folder,
    #                                                           dst_folder))
    # if rc != 0:
    #     raise Exception('copy folder error : {}'.format(err_msg))

    # copyfile(cfg.train_csv, os.path.join(args.save_path, 'train.csv'))
    # copyfile(cfg.dev_csv, os.path.join(args.save_path, 'dev.csv'))

    dataloader_train = DataLoader(
        ImageDataset(cfg.train_csv, cfg, mode='train'),
        batch_size=cfg.train_batch_size, num_workers=args.num_workers,
        drop_last=True, shuffle=True)
    dataloader_dev = DataLoader(
        ImageDataset(cfg.dev_csv, cfg, mode='dev'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False) # apatoto eituku sesh hoise 
    dev_header = dataloader_dev.dataset._label_header

    summary_train = {'epoch': 0, 'step': 0}
    summary_dev = {'loss': float('inf'), 'acc': 0.0}
    summary_writer = SummaryWriter(args.save_path)
    epoch_start = 0
    best_dict = {
        "acc_dev_best": 0.0,
        "auc_dev_best": 0.0,
        "loss_dev_best": float('inf'),
        "fused_dev_best": 0.0,
        "best_idx": 1}
    '''Needs to be moidified'''
    if args.resume:
        #ckpt_path_local = os.path.join(args.save_path, 'train.ckpt')
        ckpt_path = '/content/drive/MyDrive/learning_chexpert/best_local1.ckpt'
        ckpt = torch.load(ckpt_path, map_location=device)
        model_local.module.load_state_dict(ckpt['state_dict'])
        summary_train = {'epoch': ckpt['epoch'], 'step': ckpt['step']}
        best_dict['acc_dev_best'] = ckpt['acc_dev_best']
        best_dict['loss_dev_best'] = ckpt['loss_dev_best']
        best_dict['auc_dev_best'] = ckpt['auc_dev_best']
        epoch_start = ckpt['epoch']

    for epoch in range(epoch_start, cfg.epoch):
        lr = lr_schedule(cfg.lr, cfg.lr_factor, summary_train['epoch'],
                         cfg.lr_epochs) # reducing the learning rate on 2nd epoch, is it a good choice?
        for param_group in optimizer_global.param_groups:
            param_group['lr'] = lr

        for param_group in optimizer_local.param_groups:
            param_group['lr'] = lr
        # for param_group in optimizer_fusion.param_groups:
        #     param_group['lr'] = lr

        summary_train = train_epoch(
            summary_train, summary_dev, cfg, args, model_global,model_local,
            dataloader_train, dataloader_dev, optimizer_global,optimizer_local,
            summary_writer, best_dict, dev_header)

        time_now = time.time()
        summary_dev, predlist, true_list = test_epoch(
            summary_dev, cfg, args, model_global,model_local, dataloader_dev) # since it is validating after every epoch train what it is necissity of cfg.test_every ?
        time_spent = time.time() - time_now

        auclist = []
        for i in range(len(cfg.num_classes)):
            y_pred = predlist[i]
            y_true = true_list[i]
            fpr, tpr, thresholds = metrics.roc_curve(
                y_true, y_pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            auclist.append(auc)
        summary_dev['auc'] = np.array(auclist)

        loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                    summary_dev['loss']))
        acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                   summary_dev['acc']))
        auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                   summary_dev['auc']))

        logging.info(
            '{}, Dev, Step : {}, Loss : {}, Acc : {}, Auc : {},'
            'Mean auc: {:.3f} ''Run Time : {:.2f} sec' .format(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                summary_train['step'],
                loss_dev_str,
                acc_dev_str,
                auc_dev_str,
                summary_dev['auc'].mean(),
                time_spent))
        print(
            '{}, Dev, Step : {}, Loss : {}, Acc : {}, Auc : {},' \
            'Mean auc: {:.3f} ''Run Time : {:.2f} sec' .format(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                summary_train['step'],
                loss_dev_str,
                acc_dev_str,
                auc_dev_str,
                summary_dev['auc'].mean(),
                time_spent))

        for t in range(len(cfg.num_classes)):
            summary_writer.add_scalar(
                'dev/loss_{}'.format(dev_header[t]), summary_dev['loss'][t],
                summary_train['step'])
            summary_writer.add_scalar(
                'dev/acc_{}'.format(dev_header[t]), summary_dev['acc'][t],
                summary_train['step'])
            summary_writer.add_scalar(
                'dev/auc_{}'.format(dev_header[t]), summary_dev['auc'][t],
                summary_train['step'])

        save_best = False

        mean_acc = summary_dev['acc'][cfg.save_index].mean()
        if mean_acc >= best_dict['acc_dev_best']:
            best_dict['acc_dev_best'] = mean_acc
            if cfg.best_target == 'acc':
                save_best = True

        mean_auc = summary_dev['auc'][cfg.save_index].mean()
        if mean_auc >= best_dict['auc_dev_best']:
            best_dict['auc_dev_best'] = mean_auc
            if cfg.best_target == 'auc':
                save_best = True

        mean_loss = summary_dev['loss'][cfg.save_index].mean()
        if mean_loss <= best_dict['loss_dev_best']:
            best_dict['loss_dev_best'] = mean_loss
            if cfg.best_target == 'loss':
                save_best = True

        if save_best:
            torch.save(
                {'epoch': summary_train['epoch'],
                 'step': summary_train['step'],
                 'acc_dev_best': best_dict['acc_dev_best'],
                 'auc_dev_best': best_dict['auc_dev_best'],
                 'loss_dev_best': best_dict['loss_dev_best'],
                 'state_dict': model_local.module.state_dict()},
                os.path.join('/content/drive/MyDrive/learning_chexpert',
                             'best_local{}.ckpt'.format(best_dict['best_idx']))
            )
            best_dict['best_idx'] += 1
            if best_dict['best_idx'] > cfg.save_top_k:
                best_dict['best_idx'] = 1 # 3 taa file er moddhe konta best result er eita bujhar upay last-modified dekhe
            logging.info(
                '{}, Best, Step : {}, Loss : {}, Acc : {},'
                'Auc :{},Best Auc : {:.3f}' .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary_train['step'],
                    loss_dev_str,
                    acc_dev_str,
                    auc_dev_str,
                    best_dict['auc_dev_best']))
            print(
                '{}, Best, Step : {}, Loss : {}, Acc : {},'
                'Auc :{},Best Auc : {:.3f}' .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary_train['step'],
                    loss_dev_str,
                    acc_dev_str,
                    auc_dev_str,
                    best_dict['auc_dev_best']))
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'acc_dev_best': best_dict['acc_dev_best'],
                    'auc_dev_best': best_dict['auc_dev_best'],
                    'loss_dev_best': best_dict['loss_dev_best'],
                    'state_dict': model_local.module.state_dict()},
                   os.path.join("/content/drive/MyDrive/learning_chexpert", 'train_local.ckpt')) # saves the model after every epoch by same name , this can be used for resume training
    summary_writer.close()


def main():
    args = parser.parse_args()
    if args.verbose is True:
        print('Using the specified args:')
        print(args) #showing the all specified arguments since verbose is set to true

    run(args)


if __name__ == '__main__':
    main()
