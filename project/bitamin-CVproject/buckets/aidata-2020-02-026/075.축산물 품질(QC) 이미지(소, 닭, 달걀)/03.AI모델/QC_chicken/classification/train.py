import numpy as np
# MKL_SERVICE_FORCE_INTEL=1
import sys

import os
import os.path as osp
import torch
import torchvision
import time

import pandas as pd
import copy
import argparse
import random
import shutil
import warnings
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data

import chicken_dataset as dataset_


parser=argparse.ArgumentParser(description='training pipeline')

parser.add_argument('--log_path', default='./log', type=str)  # 로그 텍스트를 저장할 위치
parser.add_argument('--gpu', type=str, default=0, help='gpu allocation')  # 사용할 GPU 선택

parser.add_argument('--invert', default=True, help='using inverted mse loss for anomaly')  # Loss inversion 사용 여부
parser.add_argument('--FGSM', help='whether to use FGSM attack')  # FGSM Attack 사용 여부 -> Attack 된 이미지 = Abnormal 로 간주됨/

parser.add_argument('--ld', default=2, type=float, help='weight for inverted loss')  # Loss inversion 시, inversion된 loss 의 weight

parser.add_argument('--model_name', default='RESNET50')  # 사용할 모델 선택
parser.add_argument('--exp', type=str, help='model explanation', required=True)  # 훈련 방식 메모

parser.add_argument('--resize', default=448, type=int)  # 이미지 크기 재설정
parser.add_argument('--num_workers', default=6, type=int)  # 훈련에 사용할 CPU 코어 수
parser.add_argument('--esct', type=int, default=20, help='early stop count')  # early stopping 기준

parser.add_argument('--epochs', default=200, type=int)  # 전체 훈련 epoch
parser.add_argument('--batch_size', default=24, type=int)  # 배치 사이즈
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float)  # learning rate
parser.add_argument('--momentum', default=0.9, type=float)  # optimizer의 momentum
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)  # 가중치 정규화

parser.add_argument('--optim', default='SGD')  # optimizer


args=parser.parse_args()



# make folders
if not os.path.exists('./log'):
    os.mkdir('./log')
    
if not os.path.exists('./model_weight'):
    os.mkdir('./model_weight')
    
if not os.path.exists('./checkpoint'):
    os.mkdir('./checkpoint')

        
        
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    
def log(message):
    with open(osp.join(args.log_path, args.model_name)+'_'+f"{args.exp}"+'.txt', 'a+') as logger:
        logger.write(f'{message}\n')
        
        
if args.FGSM:
    import Utility_FGSM as U
    log(f'FGSM attack: {args.FGSM}')
elif args.invert:
    import Utility_invert as U
    log(f'Inverted MSE: {args.invert}')
        
def main_worker(args):    
    log(f'model name: {args.model_name}')
    log(f'Explanation: {args.exp}')
    log(f'num_workers: {args.num_workers}')
    log(f'n_epochs: {args.epochs}')
    log(f'batch_size: {args.batch_size}')    
    log(f"Resize: {args.resize}")
    log(f'Early Stop Count: {args.esct}')
    log(f"Lambda: {args.ld}")

    
    models=U.Create_Models(base_model=args.model_name, device=device)
              
    criterions = {
    "Reconstruction": nn.MSELoss(reduction='none'),
}    
    if args.optim=='SGD':
        optimizers = {
            # Encoder
            "Encoder": optim.SGD(models["Encoder"].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4),
            # Encoder with regularization for Generator loss
            "Encoder_reg": optim.SGD(models["Encoder"].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4),
            # Decoder
            "Decoder": optim.SGD(models["Decoder"].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4),
            # Discriminator
            "Discriminator": optim.SGD(models["Discriminator"].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4),
        }
    else:             
        optimizers = {
            # Encoder
            "Encoder": optim.Adam(models["Encoder"].parameters(), lr=0.0001, betas=(0.5, 0.9)),
            # Encoder with regularization for Generator loss
            "Encoder_reg": optim.Adam(models["Encoder"].parameters(), lr=0.0001),
            # Decoder
            "Decoder": optim.Adam(models["Decoder"].parameters(), lr=0.0001, betas=(0.5, 0.9)),
            # Discriminator
            "Discriminator": optim.Adam(models["Discriminator"].parameters(), lr=0.0001, betas=(0.5, 0.9)) 
        }        
        
    if args.optim=='SGD':
        encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizers["Encoder"], patience=5)
        decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizers["Decoder"], patience=5)
        discriminator_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizers["Discriminator"], patience=5)
        generator_schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizers["Encoder_reg"], patience=5)
               

    
    args.start_epoch=0
    
    
    # train transforms
    train_compose=A.Compose([
        A.Resize(args.resize, args.resize),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.RandomRotate90(p=0.4),
        A.RandomGridShuffle(p=0.4),
        ToTensorV2()
    ])
    log(train_compose)
    
    # validation transforms
    valid_compose=A.Compose([
        A.Resize(args.resize,args.resize),
        ToTensorV2()    
    ])    
    
    
    
    
    # dataset

    base_root='../Dataset/preprocessed/classification'
    train_dataset=dataset_.ChickenDataset(base_PATH=osp.join(base_root, 'train'), transforms=train_compose)
    valid_dataset=dataset_.ChickenDataset(base_PATH=osp.join(base_root, 'val'), transforms=valid_compose)
    test_dataset=dataset_.ChickenDataset(base_PATH=osp.join(base_root, 'test'), transforms=valid_compose)
    
    log(f'\ntrain size : {len(train_dataset)}')
    log(f'valid size : {len(valid_dataset)}')
    log(f'test size : {len(test_dataset)}\n')

      
    train_loader=torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    
    val_loader=torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)


    test_loader=torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    
    ## Training loop
    # train and validation, test step
    best_loss=np.inf
    best_val_f1=0.0
    
    early_stop_count=0
    lr_changed=False
    previous_lr=optimizers['Encoder'].state_dict()['param_groups'][0]['lr']
    
    from_=time.time()    
    
    #-----------------------------------------------------------------------------------------------------------
    
    for epoch in range(args.start_epoch, args.epochs):  # start_epoch 
        
        

        log(f'##------Epoch {epoch+1}')
        
        since=time.time()
        # train for one epoch
        epoch_loss, disc_loss, gen_loss=U.Train(train_loader, models, criterions, optimizers, device=device, log=log, args=args)
        
        if (args.optim=='SGD'):
            encoder_scheduler.step(epoch_loss)
            decoder_scheduler.step(epoch_loss)
            discriminator_scheduler.step(disc_loss)
            generator_schedular.step(gen_loss)
        
        
        # evaluate on validation set
        val_loss, val_f1, val_best_f1_threshold=U.Validation(val_loader, models, criterions, device=device, log=log, args=args)
        
#         diff=np.abs(acc-val_acc)
#         is_min_diff=min_diff>diff
#         min_diff=min(min_diff, diff)
        U.Test(test_loader, models, criterions, device=device, log=log, args=args, best_f1_threshold=val_best_f1_threshold)
   
        
        is_best_f1=best_val_f1<val_f1
        best_val_f1=max(best_val_f1, val_f1)
        
        is_best=best_loss>val_loss
        best_loss=min(best_loss, val_loss)
        
        save_checkpoint({
            'epoch': epoch+1,
            'arch': args.model_name,
            'state_dict': {key: models[key].state_dict() for key in models if key !='Z_dim'},
            'best_val_loss': best_loss,
            'best_val_f1' : best_val_f1,
            'val_best_f1_threshold':val_best_f1_threshold,
        }, is_best, is_best_f1)
        
        end=time.time()
        
        if is_best:
            log('\n---- Best Val Loss ----')
            
        if is_best_f1:
            log('\n---- Best Val F1-Score')
            
        log(f'\nRunning Time: {int((end-since)//60)}m {int((end-since)%60)}s\n\n')
        
        # early stopping
        #if lr_changed:
        if is_best:
            early_stop_count=0
        else:
            early_stop_count+=1
            
        log(f'\nEarly_stop_count: {early_stop_count}')
        if early_stop_count==args.esct:
            log(f'\nEarly Stopped because Validation Loss is not decreasing for {args.esct} epochs')
            break      
            
        
        
    to_=time.time()
    log(f'\nTotal Running Time: {int((to_-from_)//60)}m {int((to_-from_)%60)}s')
    #-----------------------------------------------------------------------------------------------------------    




def calculate_scores(tot_labels, tot_pred_labels):
    f1=f1_score(tot_labels, tot_pred_labels, average='macro')
    re=recall_score(tot_labels, tot_pred_labels, average='macro')
    pre=precision_score(tot_labels, tot_pred_labels, average='macro', zero_division=0)
    
    return f1, re, pre




def save_checkpoint(state, is_best, is_best_f1, filename='./checkpoint/'+args.model_name+'_'+args.exp+'.pth'):    
    
    torch.save(state, filename)
    
    if is_best:
        shutil.copyfile(filename, './model_weight/'+args.model_name+'_'+args.exp+'_best_Loss.pth')
        
    if is_best_f1:
        shutil.copyfile(filename, './model_weight/'+args.model_name+'_'+args.exp+'_best_F1.pth')



    
if __name__=='__main__':
    main_worker(args)
    
    