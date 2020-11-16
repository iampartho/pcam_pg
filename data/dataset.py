import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from data.imgaug import GetTransforms
from data.utils import transform
np.random.seed(0)


class ImageDataset(Dataset):
    def __init__(self, label_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = [] # it's a list of list each index refers to a single image label
        self._mode = mode
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]] # working with the 5 classes not 14
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_path = fields[0]
                flg_enhance = False # what is this shit ?
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8: # targeting class header[10] and header[13]
                        labels.append(self.dict[1].get(value))
                        if self.dict[1].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0: # cfg.enhance_index = [2,6]
                            flg_enhance = True
                    elif index == 2 or index == 6 or index == 10: # targeting class header[7], header[11] and header[15]
                        labels.append(self.dict[0].get(value))
                        if self.dict[0].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                # labels = ([self.dict.get(n, n) for n in fields[5:]])
                labels = list(map(int, labels)) # converting the labels to integer
                self._image_paths.append(image_path)
                assert os.path.exists(image_path), image_path
                self._labels.append(labels)
                if flg_enhance and self._mode == 'train':
                    for i in range(self.cfg.enhance_times):
                        self._image_paths.append(image_path) # why taking the same image twince in the training set
                        self._labels.append(labels)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0) # reading images in grayscale
        image = Image.fromarray(image) # converting it to pil image
        if self._mode == 'train': # No TTA used
            image = GetTransforms(image, type=self.cfg.use_transforms_type)
        image = np.array(image)
        image = transform(image, self.cfg) # this is happening for test time as well
        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))