# -*- coding: utf-8 -*-
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from other_models import U_Net,R2U_Net,AttU_Net,AttResU_Net,R2AttU_Net
from options import args
from utils import mkdir, build_dataset  # build_model,
from first_stage import SRF_UNet
# from losses import build_loss
from val import val_first_stage
from test import test_first_stage

import os
import torch

from utils import mkdir, get_lr, adjust_lr


def train_first_stage(writer, dataloader, net, optimizer, base_lr, criterion, device, power,
                      epoch, num_epochs=100):
    dt_size = len(dataloader.dataset)
    epoch_loss = 0
    step = 0
    for sample in dataloader:
        step += 1
        img = sample[0].to(device)
        gt = sample[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        pred = net(img)
        loss = criterion(pred, gt)  # 可加权
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # 当前batch图像的loss
        niter = epoch * len(dataloader) + step
        writer.add_scalars("train_loss", {"train_loss": loss.item()}, niter)
        print("%d / %d, train loss: %0.4f" % (step, (dt_size - 1) // dataloader.batch_size + 1, loss.item()))

        # 写入当前lr
        current_lr = get_lr(optimizer)
        writer.add_scalars("learning_rate", {"lr": current_lr}, niter)

    print("epoch %d loss: %0.4f" % (epoch, epoch_loss))
    print("current learning rate: %f" % current_lr)

    adjust_lr(optimizer, base_lr, epoch, num_epochs, power=power)

    return net


# 是否使用cuda
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.mode == "train":
    isTraining = True
else:
    isTraining = False

database = build_dataset(args.dataset, args.data_dir, channel=args.input_nc, isTraining=isTraining,
                         crop_size=(args.crop_size, args.crop_size), scale_size=(args.scale_size, args.scale_size))
sub_dir = args.dataset
val_database = build_dataset(args.dataset, args.val_data_dir, channel=args.input_nc, isTraining=False,
                             crop_size=(args.crop_size, args.crop_size), scale_size=(args.scale_size, args.scale_size))
val_dataloader = DataLoader(val_database, batch_size=1)
if isTraining:  # train
    writer = SummaryWriter(args.logs_dir + "/" + sub_dir)
    mkdir(args.models_dir + "/" + sub_dir)  # two stage时可以创建first_stage和second_stage这两个子文件夹
    if args.backbone== "res_unet":
        args.batch_size=8
    elif args.backbone== "att_unet":
        args.batch_size=8
    elif args.backbone== "ce_net":
        args.batch_size=8
    elif args.backbone== "cs_net":
        args.batch_size=8
    elif args.backbone == "srf_unet":
        args.batch_size = 8
    # 加载数据集
    train_dataloader = DataLoader(database, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    if args.backbone== "unet":
        first_net = U_Net(img_ch=args.input_nc, output_ch=1).to(device)
    elif args.backbone== "res_unet":
        first_net = R2U_Net(img_ch=args.input_nc, output_ch=1).to(device)
    elif args.backbone== "att_unet":
        first_net = AttU_Net(img_ch=args.input_nc, output_ch=1).to(device)
    elif args.backbone== "ce_net":
        first_net = AttResU_Net(img_ch=args.input_nc, output_ch=1).to(device)
    elif args.backbone== "cs_net":
        first_net = R2AttU_Net(img_ch=args.input_nc, output_ch=1).to(device)
    elif args.backbone== "srf_unet":
        first_net = SRF_UNet(img_ch=args.input_nc, output_ch=1).to(device)

    first_optim = optim.Adam(first_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    
    criterion = torch.nn.MSELoss()  # 可更改
    # start training
    print("Start training...")
    for epoch in range(args.first_epochs):
        print('Epoch %d / %d' % (epoch + 1, args.first_epochs))
        print('-'*10)
        first_net = train_first_stage(writer, train_dataloader, first_net, first_optim, args.init_lr, criterion, device, args.power, epoch, args.first_epochs)
        if (epoch + 1) % args.val_epoch_freq == 0 or epoch == args.first_epochs - 1:
            first_net = val_first_stage(val_dataloader, first_net, device,
                                                    args.save_epoch_freq, args.models_dir + "/" + sub_dir+"/"+ args.backbone,
                                                    args.results_dir + "/" + sub_dir+ "/"+ args.backbone, epoch, args.first_epochs)
    print("Training finished.")
