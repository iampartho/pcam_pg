import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from data.utils import transform # noqa
from model.utils import tensor2numpy # noqa
from skimage.measure import label

disease_classes = [
    'Cardiomegaly',
    'Edema',
    'Consolidation',
    'Atelectasis',
    'Pleural Effusion'
]


def fig2data(fig):
    fig.canvas.draw()
    # Get the RGB buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    buf = buf[:, :, (2, 1, 0)]
    return np.ascontiguousarray(buf)


class Heatmaper(object):
    def __init__(self, alpha, prefix, cfg, model, device):
        super(Heatmaper, self).__init__()
        # init cfg
        self.cfg = cfg
        # init device
        self.device = device
        # init model
        self.model = model
        # init transparency
        self.alpha = alpha
        # init filename prefix
        self.prefix = prefix

    def image_reader(self, image_file):
        """
        Args:
            image_file: str to an image file path
        Returns:
            image: (1, C, H, W) tensor of image
            image_color: (H, W, C) numpy array of RGB image
        """
        image_gray = cv2.imread(image_file, 0)
        assert image_gray is not None, "invalid image read in: {}"\
            .format(image_file)
        image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        image = transform(image_gray, self.cfg)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        return image, image_color

    def set_overlay(self, figure, row_, i, subtitle):
        ax_overlay = figure.add_subplot(row_, 3, 2 + i)
        ax_overlay.set_title(subtitle, fontsize=10, color='r')
        ax_overlay.set_yticklabels([])
        ax_overlay.set_xticklabels([])
        ax_overlay.tick_params(axis='both', which='both', length=0)
        return ax_overlay

    def set_rawimage(self, figure, row_):
        ax_raw_image = figure.add_subplot(row_, 3, 1)
        subtitle = 'original image'
        ax_raw_image.set_title(subtitle, fontsize=10, color='r')
        ax_raw_image.set_yticklabels([])
        ax_raw_image.set_xticklabels([])
        ax_raw_image.tick_params(axis='both', which='both', length=0)
        return ax_raw_image

    def get_raw_image(self, image, ax):
        """
        Args:
            image: (H, W, C) numpy array of BGR image
            ax: AxesSubplot of matplotlib.axes
        Returns:
            image: AxesImage of matplotlib.image
        """
        image = ax.imshow(image)
        return image

    def get_overlayed_smooth(self, ax, prob_map):
        long_side = self.cfg.long_side
        image = ax.imshow(cv2.resize(prob_map, (long_side, long_side)),
                          cmap='jet',
                          vmin=0.0,
                          vmax=1.0,
                          alpha=self.alpha)
        return image

    def get_overlayed_img(self, ori_image, logit_map,
                          prob_map, ax):
        """
        Args:
            ori_image: (H, W) numpy array of gray image
            logit_map: (H, W) numpy array of model prediction
            prob_map: (H, W) numpy array of model prediction with prob
            ax: AxesSubplot of matplotlib.axes
        Returns:
            image: AxesImage of matplotlib.image
        """
        overlayed_image = ax.imshow(ori_image, cmap='gray', vmin=0, vmax=255)
        overlayed_image = self.get_overlayed_smooth(ax, prob_map)

        return overlayed_image

    def gen_heatmap(self, image_file):
        """
        Args:
            image_file: str to a jpg file path
        Returns:
            prefix_name: str of a prefix_name of a jpg with/without prob
            figure_data: numpy array of a color image
        """
        image_tensor, image_color = self.image_reader(image_file)
        image_tensor = image_tensor.to(self.device)
        # model inference
        logits, logit_maps = self.model(image_tensor)
        logits = torch.stack(logits)
        logit_maps = torch.stack(logit_maps)
        # tensor to numpy
        image_np = tensor2numpy(image_tensor)
        prob_maps_np = tensor2numpy(torch.sigmoid(logit_maps))
        logit_maps_np = tensor2numpy(logit_maps)

        cropped_image = get_crop_image(image_file, prob_maps_np)

        num_tasks = len(disease_classes)
        row_ = num_tasks // 3 + 1
        plt_fig = plt.figure(figsize=(10, row_*4), dpi=300)
        prefix = -1 if self.prefix is None else \
            disease_classes.index(self.prefix)
        prefix_name = ''
        # vgg and resnet do not use pixel_std, densenet and inception use.
        ori_image = image_np[0, 0, :, :] * self.cfg.pixel_std + \
            self.cfg.pixel_mean
        for i in range(num_tasks):
            prob = torch.sigmoid(logits[i])
            prob = tensor2numpy(prob)
            if prefix == i:
                prefix_name = '{:.4f}_'.format(prob[0][0])
            subtitle = '{}:{:.4f}'.format(disease_classes[i],
                                          prob[0][0])
            ax_overlay = self.set_overlay(plt_fig, row_, i, subtitle)
            # overlay_image is assigned multiple times,
            # but we only use the last one to get colorbar
            overlay_image = self.get_overlayed_img(ori_image,
                                                   logit_maps_np[i, :, :],
                                                   prob_maps_np[i, :, :],
                                                   ax_overlay)
        ax_rawimage = self.set_rawimage(plt_fig, row_)
        _ = self.get_raw_image(image_color, ax_rawimage)
        divider = make_axes_locatable(ax_overlay)
        ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
        plt_fig.colorbar(overlay_image, cax=ax_colorbar)
        plt_fig.tight_layout()
        figure_data = fig2data(plt_fig)
        plt.close()
        return prefix_name, figure_data,cropped_image




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





def get_crop_image(ori_image_path, feature_conv):
    image = cv2.imread(ori_image_path)
    size_upsample = (256, 256) 
    all_idx=np.array([])
    cls_number, h, w = feature_conv.shape
    for i in range(cls_number):
        feature = feature_conv[i, :, :]
        # cam = feature.reshape((nc, h*w))
        # cam = cam.sum(axis=0)
        cam = feature.reshape(h,w)
        #cam = cam - np.min(cam)
        #cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam)

        heatmap_bin = binImage(cv2.resize(cam_img, size_upsample))
        heatmap_maxconn = selectMaxConnect(heatmap_bin)
        heatmap_mask = heatmap_bin * heatmap_maxconn
        #print(heatmap_mask)
        
        ind = np.argwhere(heatmap_mask != 0)
        if i==0:
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
    #image = ori_image[i].numpy().reshape(256,256,3)
    #image = image[int(256*0.334):int(256*0.667),int(256*0.334):int(256*0.667),:]

    
    image_crop = image[minh:maxh,minw:maxw,:] # because image was normalized before
    image_crop = cv2.resize(image_crop, size_upsample)
    return image_crop