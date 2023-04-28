# -*- coding: utf-8 -*-

import os
from utils import mkdir

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
# from fundus_dataset import recompone_overlap
from evaluation import *

from copy import deepcopy
def get_results(dataloader, results_dir, pred, isSave=True):

    pred_arr = pred.squeeze().cpu().numpy()

    pred_img = np.array(pred_arr * 255, np.uint8)
    #
    # thresh_value, thresh_pred_img = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = deepcopy(pred_arr)
    binary[binary>=0.5]=1
    binary[binary<0.5]=0
    thresh_pred_img = np.array(binary * 255, np.uint8)

    print("shape of prediction", thresh_pred_img.shape)
    # Save Results
    imgpath = dataloader.dataset.getFileName()
    hospital_id = imgpath[0]
    people_id = imgpath[1]
    img_name = imgpath[2]
    if isSave:
        mkdir(results_dir + "/" + hospital_id + "/" + people_id + "/Thresh")
        mkdir(results_dir + "/" + hospital_id + "/" + people_id + "/noThresh")
        cv2.imwrite(results_dir + "/" + hospital_id + "/" + people_id + "/noThresh/" + img_name, pred_img)
        cv2.imwrite(results_dir + "/" + hospital_id + "/" + people_id + "/Thresh/" + img_name, thresh_pred_img)

def test_first_stage(dataloader, net, device, results_dir):
    i = 1
    with torch.no_grad():
        for sample in dataloader:
            if len(sample) != 5 and len(sample) != 4 and len(sample) != 2:
                print("Error occured in sample %03d, skip" % i)
                continue

            print("Evaluate %03d..." % i)
            i += 1

            img = sample[0].to(device)
            gt = sample[1].to(device)

            pred = net(img)
            get_results(dataloader, results_dir, pred)
def val_first_stage(val_dataloader, net, device, save_epoch_freq,
                    models_dir, results_dir, epoch, num_epochs=100):
    net.eval()
    test_first_stage(val_dataloader, net, device, results_dir)
    # 保存模型
    mkdir(models_dir)
    checkpoint_path = os.path.join(models_dir, "{net}-{epoch}.pth")
    if (epoch + 1) % save_epoch_freq == 0:
        torch.save(net, checkpoint_path.format(net="front_model", epoch=epoch+1))
    if epoch == num_epochs - 1:
        torch.save(net, os.path.join(models_dir, "front_model-latest.pth"))
    net.train(mode=True)
    
    return net

