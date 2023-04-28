###################################################
#
#   Script to pre-process the original imgs
#
##################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt


#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================

#convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

#==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized
import cv2,imageio,PIL
from PIL import Image
def readImg(im_fn):
    if "images" in im_fn:
        if "PRIME-FP20_DataPort" in im_fn:
            img = PIL.Image.open(im_fn).convert('RGB')
            img = img.resize((2600, 3000), Image.BICUBIC)
        else:
            img = PIL.Image.open(im_fn).convert('RGB')
    elif "labels" or "FOVs" in im_fn:
        if "PRIME-FP20_DataPort" in im_fn:
            img = PIL.Image.open(im_fn)
            img = img.resize((2600, 3000), Image.NEAREST)
        else:
            img = PIL.Image.open(im_fn)
    return img

if __name__ == '__main__':
    # test network forward
    import matplotlib.pyplot as plt  # plt 用于显示图片

    import matplotlib.image as mpimg  # mpimg 用于读取图片

    import numpy as np

    # save_path = '/mnt/workdir/fengwei/20210412_qinghuachanggeng_AMD-2/octa_master/dataset/data/ROSE_all/rose1/outer retina_1/gt/OR_1-2_1.png'
    # img_list = '/mnt/workdir/fengwei/20210412_qinghuachanggeng_AMD-2/octa_master/dataset/data/ROSE_all/rose1/outer retina_2/gt/OR_1-2.png'
    import glob

    normal_path = sorted(glob.glob('/mnt/workdir/fengwei/20210412_qinghuachanggeng_AMD-2/octa_master/dataset/data/ROSE_all/test_seg/T cho/gt/*.png'))
    mask_list = '/mnt/workdir/fengwei/20210412_qinghuachanggeng_AMD-2/octa_master/code/OCTA-Net/results/rose1/10666814/noThresh/10666814_xin_shu fang__1513_HD Angio Retina_OD_2020-07-28_14-17-48_F_1951-11-19_Enface-400x400-Outer Retina.png'
    # save_path = '/mnt/workdir/fengwei/vo_huaxi/final/dauda_image/dauda_image-master-tmp/mask_tmp_img/'
    mask = cv2.imread(mask_list)
    for i in range(len(normal_path)):
        img_list = normal_path[i]
        img = cv2.imread(img_list)
        # mask = mpimg.imread(mask_list).astype(np.uint8)
        # mask = cv2.imread(mask_list)
        # img_a = img[:,:,2]
        # thresh_value, img_a = cv2.threshold(img_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img[img!=0] =255
        img[:,:, 0] = img[:,:,2]
        img[:, :, 1] = img[:,:,2]
        save_path = img_list
            # .replace('.png','.jpg')
        cv2.imwrite(save_path, img)
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from skimage.io import imread, imshow
    # from skimage.draw import disk
    # # from skimage.morphology import (erosion, dilation,
    # #                                 closed, opening, area_looking, area_opening)
    # # from skimage.color import rgb2gray
    #
    # element = np.array([[0, 1, 0],
    #                     [1, 1, 1],
    #                     [0, 1, 0]])
    # plt.imshow(element, cmap='gray');
    # circle_image = np.zeros((25, 40))
    # circle_image[disk((12, 12), 8)] = 1
    # circle_image[disk((12, 28), 8)] = 1
    # for x in range(20):
    #    circle_image[np.random.randint(25), np.random.randint(40)] = 1
    # imshow(circle_image);