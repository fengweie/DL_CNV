# # -*- coding: utf-8 -*-
#
# import os
# import cv2
import argparse
import glob
# # -*- coding: utf-8 -*-
# import math
# import numpy as np
# from sklearn import metrics
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score
#
# def max_fusion(x, y):
#     assert x.shape == y.shape
#
#     return np.maximum(x, y)
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--prob_dir", type=str,
#                     default="/mnt/workdir/fengwei/20210412_qinghuachanggeng_AMD-2/octa_master/code/OCTA-Net/results/rose1/*/img/noThresh/*.png",
#                      help="path to folder for saving probability maps")
# parser.add_argument("--pred_dir", type=str,
#                     default="/mnt/workdir/fengwei/20210412_qinghuachanggeng_AMD-2/octa_master/code/OCTA-Net/results/rose1/*/img/Thresh/*.png",
#                     help="path to folder for saving prediction maps")
# parser.add_argument("--gt_dir", type=str,
#                     default="/mnt/workdir/fengwei/20210412_qinghuachanggeng_AMD-2/octa_master/dataset/data/ROSE_all/test_seg/*/gt/*.png",
#                     help="path to folder for saving ground truth")
# args = parser.parse_args()
#
# prob_lst = sorted(glob.glob(args.prob_dir))
# print(len(prob_lst))
# pred_lst = sorted(glob.glob(args.pred_dir))
# print(len(pred_lst))
# gt_lst = sorted(glob.glob(args.gt_dir))
# print(len(gt_lst))
# assert len(prob_lst) == len(pred_lst) and len(pred_lst) == len(gt_lst)
# prob_vec_all = []
# pred_vec_all = []
# gt_vec_all = []
#
# for i in range(len(pred_lst)):
#     prob_arr = cv2.imread(prob_lst[i], 0) / 255.0
#     pred_arr = cv2.imread(pred_lst[i], 0) // 255
#     gt_arr = cv2.imread(gt_lst[i], 0) // 255
#
#     prob_vec = prob_arr.flatten()
#     pred_vec = pred_arr.flatten()
#     gt_vec = gt_arr.flatten()
#     prob_vec_all.extend(prob_vec)
#     pred_vec_all.extend(pred_vec)
#     gt_vec_all.extend(gt_vec)
# roc_auc = metrics.roc_auc_score(gt_vec_all, prob_vec_all)
# print("====>roc_auc:" +str(round(roc_auc,4)))
# confusion = confusion_matrix(gt_vec_all, pred_vec_all)
# print(confusion)
# TN = confusion[0, 0]
# TP = confusion[1, 1]
# FP = confusion[0, 1]
# FN = confusion[1, 0]
# accuracy = 0
# if float(np.sum(confusion)) != 0:
#     accuracy = float( TP+TN ) / float(np.sum(confusion))
# print("Global Accuracy: " + str(round(accuracy,4)))
# specificity = 0
# if float(confusion[0, 0] + confusion[0, 1]) != 0:
#     specificity = float(TN) / float(TN + FP)
# print("Specificity: " + str(round(specificity,4)))
# sensitivity = 0
# if float(confusion[1, 1] + confusion[1, 0]) != 0:
#     sensitivity = float(TP) / float(TP + FN)
# print("Sensitivity: " + str(round(sensitivity,4)))
# precision = 0
# if float(confusion[1, 1] + confusion[0, 1]) != 0:
#     precision = float(TP) / float(TP + FP)
# print("Precision: " + str(round(precision,4)))
#
# gmean = math.sqrt(sensitivity * specificity)
# print("====>gmean:" +str(round(gmean,4)))
#
# matrix = np.array([[TP, FP],
#                    [FN, TN]])
# n = np.sum(matrix)
# sum_po = 0
# sum_pe = 0
# for i in range(len(matrix[0])):
#     sum_po += matrix[i][i]
#     row = np.sum(matrix[i, :])
#     col = np.sum(matrix[:, i])
#     sum_pe += row * col
# po = sum_po / n
# pe = sum_pe / (n * n)
# # print(po, pe)
# kappa = (po - pe) / (1 - pe)
# print("====>kappa:" +str(round(kappa,4)))
# fdr = FP / (FP + TP + 1e-12)
# print("====>fdr:" +str(round(fdr,4)))
# iou = TP / (FP + FN + TP + 1e-12)
# print("====>iou:" +str(round(iou,4)))
# dice = 2.0 * TP / (FP + FN + 2.0 * TP + 1e-12)
# print("====>dice:" +str(round(dice,4)))
import os
import cv2
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

bwith = 0.4
fig = plt.figure(figsize=(8, 8))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

ax = fig.add_subplot(1, 1, 1)
# ax = plt.gca()
# ax.set_title("ROSE-1", fontsize=20)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.grid(linestyle='-.', linewidth=0.05)

parser = argparse.ArgumentParser()
parser.add_argument("--prob_dir", type=str,
                    default="/mnt/workdir/fengwei/20210412_qinghuachanggeng_AMD-2/octa_master/code/OCTA-Net/results/rose_wo/unet/*/img/noThresh/*.png",
                     help="path to folder for saving probability maps")
parser.add_argument("--pred_dir", type=str,
                    default="/mnt/workdir/fengwei/20210412_qinghuachanggeng_AMD-2/octa_master/code/OCTA-Net/results/rose_wo/unet/*/img/Thresh/*.png",
                    help="path to folder for saving prediction maps")
parser.add_argument("--gt_dir", type=str,
                    default="/mnt/workdir/fengwei/20210412_qinghuachanggeng_AMD-2/octa_master/dataset/data/ROSE_all/test_seg/*/gt/*.png",
                    help="path to folder for saving ground truth")
args = parser.parse_args()

prob_lst = sorted(glob.glob(args.prob_dir))
print(len(prob_lst))
pred_lst = sorted(glob.glob(args.pred_dir))
print(len(pred_lst))
gt_lst = sorted(glob.glob(args.gt_dir))
print(len(gt_lst))
assert len(prob_lst) == len(pred_lst) and len(pred_lst) == len(gt_lst)
prob_vec_all = []
pred_vec_all = []
gt_vec_all = []

for i in range(len(pred_lst)):
    prob_arr = cv2.imread(prob_lst[i], 0) / 255.0
    pred_arr = cv2.imread(pred_lst[i], 0) // 255
    gt_arr = cv2.imread(gt_lst[i], 0) // 255

    prob_vec = prob_arr.flatten()
    pred_vec = pred_arr.flatten()
    gt_vec = gt_arr.flatten()
    prob_vec_all.extend(prob_vec)
    pred_vec_all.extend(pred_vec)
    gt_vec_all.extend(gt_vec)

fpr, tpr, thresholds = metrics.roc_curve(gt_vec_all, prob_vec_all, pos_label=1)
roc_auc = metrics.roc_auc_score(gt_vec_all, prob_vec_all)
plt.plot(fpr, tpr, label=" (AUC={0:.4f})".format(roc_auc),
         linewidth=2, markersize=1)

font = {'family': 'Liberation Sans',
        'weight': 'normal',
        'size': 14}
font1 = {'family': 'Liberation Sans',
         'weight': 'normal',
         'size': 20}
plt.xlim(0, 1)
plt.ylim(0, 1)
# plt.xticks(np.linspace(0, 1, 6))
plt.yticks(np.linspace(0., 1, 11))
plt.xlabel('1-Specificity', font1)
plt.ylabel('Sensitivity', font1)
plt.xticks(fontproperties='Liberation Sans', size=12, weight='normal')
plt.yticks(fontproperties='Liberation Sans', size=12, weight='normal')

plt.legend(loc='lower right', prop=font)
# plt.savefig('./figures/rosea.eps')
plt.savefig('roc_auc.png')
# plt.show()

# ##################################################################################################