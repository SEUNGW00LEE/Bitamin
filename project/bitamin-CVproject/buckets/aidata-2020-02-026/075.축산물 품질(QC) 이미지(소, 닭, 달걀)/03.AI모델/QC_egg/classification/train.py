import numpy as np
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
import torch.nn.functional as F


model_names= sorted(name for name in models.__dict__
                   if name.islower() and not name.startswith("__")
                    and callable(models.__dict__[name]))
model_names.extend(['eff','eff_b7'])


parser=argparse.ArgumentParser(description='training pipeline for image classification')

parser.add_argument('--log_path', default='./log', type=str)  # 로그 텍스트를 저장할 위치

parser.add_argument('--gpu', default=0, help='gpu allocation')  # 사용할 GPU 선택

parser.add_argument('--h', dest='height',default=512, type=int, help='resize height')  # 이미지 크기 재설정 : height
parser.add_argument('-w', dest='width', default=512, type=int, help='resize width')  # 이미지 크기 재설정 : width

parser.add_argument('--esct', default=15, type=int, help='early stop count')  # early stopping 기준

parser.add_argument('--model_name', default='eff_b7', choices=model_names)  # 사용할 모델 선택
parser.add_argument('--exp', type=str, help='model explanation', required=True)  # 훈련 방식 메모

parser.add_argument('--num_workers', default=24, type=int)  # 훈련에 사용할 CPU 코어 수

parser.add_argument('--epochs', default=200, type=int)  # 전체 훈련 epoch
parser.add_argument('--batch_size', default=8, type=int)  # 배치 사이즈
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float)  # learning rate
parser.add_argument('--momentum', default=0.9, type=float)  # optimizer의 momentum
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)  # 가중치 정규화

parser.add_argument('--optim', default='SGD')  # optimizer

parser.add_argument('--pretrained', dest='pretrained', type=str, default=True,                   
                    help='use pre-trained model')  # pre-train 모델 사용 여부




args=parser.parse_args()



# make folders
if not os.path.exists('./log'):
    os.mkdir('./log')
    
if not os.path.exists('./model_weight'):
    os.mkdir('./model_weight')
    
if not os.path.exists('./checkpoint'):
    os.mkdir('./checkpoint')


def log(message):
    print(message)
    #with open(osp.join(args.log_path, args.model_name)+'_'+f"{args.exp}"+'.txt', 'a+') as logger:
    #    logger.write(f'{message}\n')

    
device=f'cuda:{args.gpu}'

def main_worker(args):
    log(f'model name: {args.model_name}')
    log(f'Explanation: {args.exp}')
    log(f'num_workers: {args.num_workers}')
    log(f'n_epochs: {args.epochs}')
    log(f'batch_size: {args.batch_size}')

    log(f'Resize- Height: {args.height}, Width: {args.width}')
    
    # model arrangement
    
    if 'eff' in str(args.model_name):
        from efficientnet_pytorch import EfficientNet
        model=EfficientNet.from_name(model_name='efficientnet-b7', num_classes=1)
    
    try:  # for efficientnet
        if args.pretrained:            
            log(f'\n=> using pre-trained model {args.model_name}')
            
            if 'eff' in str(args.model_name):
                from efficientnet_pytorch import EfficientNet
                model=EfficientNet.from_pretrained('efficientnet-b7')

                
                model._fc=nn.Sequential(
                    nn.Linear(2560, 1, bias=True))
            else:
                model=models.__dict__[args.model_name](pretrained=True)

        else:
            log(f'\n=> creating model {args.model_name}')
            model=models.__dict__[args.model_name](pretrained=False)     
    except:
        pass

    
    if 'res' in str(args.model_name):
        model.fc=nn.Linear(model.fc.in_features, 1)
        
    elif 'vgg' in str(args.model_name):
        model.classifier[-4]=nn.Linear(in_features=4096, out_features=256)
        model.classifier[-1]=nn.Linear(in_features=256, out_features=1)
        
    elif 'dense' in str(args.model_name):
        model.classifier=nn.Sequential(
            nn.Linear(model.classifier.in_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1))

   
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model=model.to(device)
    
       

    criterion=nn.BCEWithLogitsLoss().to(device)
    log(criterion)
    
    if args.optim=='SGD':
        optimizer=optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
        lr_scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

        
    log(f"optimizer: {optimizer}")
    
    args.start_epoch=0

    
    
    # train transforms
    train_compose=A.Compose([
        A.Resize(args.height, args.width),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.RandomRotate90(p=0.4),
        A.RandomGridShuffle(p=0.4),
        
        ToTensorV2()
    ])
    log(train_compose)
    
    # validation transforms
    valid_compose=A.Compose([
        A.Resize(args.height, args.width),
        ToTensorV2()    
    ])    

    import egg_dataset as dataset_
    
    base_root='../Dataset/preprocessed/classification'
    train_dataset=dataset_.EggDatasetBasic(base_PATH=osp.join(base_root, 'train'), transforms=train_compose)
    valid_dataset=dataset_.EggDatasetBasic(base_PATH=osp.join(base_root, 'val'), transforms=valid_compose)
    test_dataset=dataset_.EggDatasetBasic(base_PATH=osp.join(base_root, 'test'), transforms=valid_compose)
    
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
    best_val_acc=0.0
    best_val_f1=0.0
    best_f1_diff=1.
    
    early_stop_count=0
    lr_changed=False
    previous_lr=optimizer.state_dict()['param_groups'][0]['lr']
    
    from_=time.time()    
    
    #-----------------------------------------------------------------------------------------------------------
    
    for epoch in range(args.start_epoch, args.epochs):  # start_epoch 
        
        

        log(f'##------Epoch {epoch+1}')  
        
        
        since=time.time()
        # train for one epoch
        epoch_loss=train(train_loader, model, criterion, optimizer, epoch, args)
        
        # evaluate on validation set
        val_loss, val_acc, val_f1=validate(val_loader, model, criterion, args)
                
        test_f1=test(test_loader, model, criterion, args)

        
        if (args.optim=='SGD') and (epoch>=9):            
            lr_scheduler.step(epoch_loss)
            current_lr=optimizer.state_dict()['param_groups'][0]['lr']

            if previous_lr > current_lr:
                log(f"\n+ Learning Rate was decreased from {previous_lr} to {current_lr} +")
                previous_lr=current_lr
                lr_changed=True
            
        # remenber best and save checkpoint
        is_best_acc=best_val_acc<val_acc
        best_val_acc=max(best_val_acc, val_acc)   
        
        is_best_f1=best_val_f1<val_f1
        best_val_f1=max(best_val_f1, val_f1)
        
        is_best=best_loss>val_loss
        best_loss=min(best_loss, val_loss)
        
        current_diff=abs(val_f1-test_f1)
        is_best_diff=best_f1_diff>current_diff
        best_f1_diff=min(best_f1_diff, current_diff)
        
        save_checkpoint({
            'epoch': epoch+1,
            'arch': args.model_name,
            'state_dict': model.state_dict(),
            'best_val_loss': best_loss,
            'best_val_acc' : best_val_acc,
            'best_val_f1' : best_val_f1,
            'optimizer': optimizer.state_dict()
        }, is_best, is_best_acc, is_best_f1, is_best_diff)
        
        end=time.time()
        
        if is_best:
            log('\n---- Best Val Loss ----')
            
        if is_best_acc:
            log('\n---- Best Val Accuracy ----')
            
        if is_best_f1:
            log('\n---- Best Val F1-Score')
            
        if is_best_diff:
            log('\n---- Best val-test f1 difference')
            
        log(f'\nRunning Time: {int((end-since)//60)}m {int((end-since)%60)}s\n\n')
        
        # early stopping
        if lr_changed==True:
            if is_best_acc:
                early_stop_count=0
            else:
                early_stop_count+=1

            if early_stop_count==args.esct:
                log(f'\nEarly Stopped because Validation Acc is not increasing for {args.esct} epochs')
                break      
            
        
        
    to_=time.time()
    log(f'\nTotal Running Time: {int((to_-from_)//60)}m {int((to_-from_)%60)}s')
    #-----------------------------------------------------------------------------------------------------------    
    
# train for one epoch
def train(train_loader, model, criterion, optimizer, epoch, args):
    
    model=model.train()
    
    running_loss=0.0
    correct=0
    total=0
    
    tot_labels=[]
    tot_pred_labels=[]
    scaler = torch.cuda.amp.GradScaler()
    
    for i, (images, target) in enumerate(train_loader):
        
        target=target[target!=99]
        images=images[~images.isnan()].view(-1, 3, args.width, args.height)        
        
        images=images.to(device)
        target=target.to(device)
        
        with torch.cuda.amp.autocast():
            output=model(images)
            loss=criterion(output, target.view(-1,1))
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()       
        scaler.step(optimizer)       
        scaler.update()
        
        running_loss+=loss.item()*images.size(0)
        
        # accuracy
        #_, output_index=torch.max(output, 1)
        output_index=torch.round(torch.sigmoid(output.view(-1)))
        
        total+=target.size(0)
        correct+=(output_index==target).sum().float()  
        
        tot_labels.extend(list(target.cpu().numpy()))
        tot_pred_labels.extend(list(output_index.view(-1).detach().cpu().numpy()))       

    
    acc=100*correct/total
    
    epoch_loss=running_loss / len(train_loader.dataset)
       
    log(f"[+] Train Accuracy: {acc :.3f},  Train Loss: {epoch_loss :.4f}")
    
    f1, re, pre=calculate_scores(tot_labels, tot_pred_labels)
    
    log(f"[+]  F1: {f1 :.3f}, Precision: {pre :.3f}, ReCall: {re :.3f}\n")
    
    return epoch_loss

        

def validate(val_loader, model, criterion, args):
    
    model=model.eval()
    
    with torch.no_grad():
        
        running_loss=0.0
        total=0
        correct=0
        
        tot_labels=[]
        tot_pred_labels=[]

        for i, (images, target) in enumerate(val_loader):
                        
            target=target[target!=99]      
            images=images[~images.isnan()].view(-1, 3, args.width, args.height)        
            
            
            images=images.to(device)
            target=target.to(device)   
            
            with torch.cuda.amp.autocast():        
                output=model(images)
                loss=criterion(output, target.view(-1,1))
            
            running_loss+=loss.item()*target.size(0)
            
            #_, output_index=torch.max(output, 1)
            output_index=torch.round(torch.sigmoid(output.view(-1)))
            
            total+=target.size(0)
            correct+=(output_index==target).sum().float()
            
            tot_labels.extend(list(target.cpu().numpy()))
            tot_pred_labels.extend(list(output_index.view(-1).detach().cpu().numpy()))       
            
        acc=100*correct/total
        
        val_loss=running_loss / len(val_loader.dataset)
        
        log(f"[+] Validation Accuracy: {acc :.3f},  Val Loss: {val_loss :.4f}")
        
        f1, re, pre=calculate_scores(tot_labels, tot_pred_labels)
        log(f"[+]  F1: {f1 :.3f}, Precision: {pre :.3f}, ReCall: {re :.3f}\n")
        
    return val_loss, acc, f1


def test(test_loader, model, criterion, args):
    
    model=model.eval()
    
    with torch.no_grad():
        
        running_loss=0.0
        total=0
        correct=0
        
        tot_labels=[]
        tot_pred_labels=[]

        for i, (images, target) in enumerate(test_loader):
            
            target=target[target!=99]
            images=images[~images.isnan()].view(-1, 3, args.width, args.height)  
            
            
            images=images.to(device)
            target=target.to(device)   
            
            with torch.cuda.amp.autocast():        
                output=model(images)
                loss=criterion(output, target.view(-1,1))
            
            running_loss+=loss.item()*target.size(0)
            
            #_, output_index=torch.max(output, 1)
            output_index=torch.round(torch.sigmoid(output.view(-1)))
            
            total+=target.size(0)
            correct+=(output_index==target).sum().float()
            
            tot_labels.extend(list(target.cpu().numpy()))
            tot_pred_labels.extend(list(output_index.view(-1).cpu().numpy()))    
            
        
        acc=100*correct/total
        
        test_loss=running_loss / len(test_loader.dataset)
        
        log(f"[+] Test Accuracy: {acc :.3f},  Test Loss: {test_loss :.4f}")
        
        f1, re, pre=calculate_scores(tot_labels, tot_pred_labels)
        log(f"[+]  F1: {f1 :.3f},  Precision: {pre :.3f},  ReCall: {re :.3f}\n")
        
    return f1


def calculate_scores(tot_labels, tot_pred_labels):
    f1=f1_score(tot_labels, tot_pred_labels, average='macro')
    re=recall_score(tot_labels, tot_pred_labels, average='macro')
    pre=precision_score(tot_labels, tot_pred_labels, average='macro', zero_division=0)
    
    return f1, re, pre




def save_checkpoint(state, is_best, is_best_acc, is_best_f1, is_best_diff, filename='./checkpoint/'+args.model_name+'_'+args.exp+'.pht'):
    torch.save(state, filename)
    if is_best_acc:
        shutil.copyfile(filename, './model_weight/'+args.model_name+'_'+args.exp+'_best_Acc.pth')
    
    if is_best:
        shutil.copyfile(filename, './model_weight/'+args.model_name+'_'+args.exp+'_best_Loss.pth')
        
    if is_best_f1:
        shutil.copyfile(filename, './model_weight/'+args.model_name+'_'+args.exp+'_best_F1.pth')
        
    if is_best_diff:
        shutil.copyfile(filename, './model_weight/'+args.model_name+'_'+args.exp+'_best_f1_diff.pth')        


    
if __name__=='__main__':
    main_worker(args)
    
    