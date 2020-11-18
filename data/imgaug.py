import cv2
import torchvision.transforms as tfs
import torchvision.transforms.functional as tvf
import numpy as np

def Common(image):

    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image



def uniform(a, b):
    return a + np.random.rand() * (b-a)

def Aug(image):
    # if np.random.rand() > 0.4:
    #         img = tvf.adjust_brightness(img, uniform(0.3,1.5))
    #     if np.random.rand() > 0.7:
    #         factor = 2 ** uniform(-1, 1)
    #         img = tvf.adjust_contrast(img, factor) # 0.5 ~ 2
    #     if np.random.rand() > 0.7:
    #         img = tvf.adjust_hue(img, uniform(-0.1,0.1))
    #     if np.random.rand() > 0.6:
    #         factor = uniform(0,2)
    #         if factor > 1:
    #             factor = 1 + uniform(0, 2)
    #         img = tvf.adjust_saturation(img, factor) # 0 ~ 3
    #     if np.random.rand() > 0.5:
    #         img = tvf.adjust_gamma(img, uniform(0.5, 3))
    factor = 2 ** uniform(-1, 1)
    image = tvf.adjust_contrast(image, factor) # 0.5 ~ 2
    
    img_aug = tfs.Compose([
        tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                         scale=(0.95, 1.05), fillcolor=128),
    ])
    image = img_aug(image)

    return image


def GetTransforms(image, target=None, type='common'):
    # taget is not support now
    if target is not None:
        raise Exception(
            'Target is not support now ! ')
    # get type
    if type.strip() == 'Common':
        image = Common(image)
        return image
    elif type.strip() == 'None':
        return image
    elif type.strip() == 'Aug':
        image = Aug(image)
        return image
    else:
        raise Exception(
            'Unknown transforms_type : '.format(type))
