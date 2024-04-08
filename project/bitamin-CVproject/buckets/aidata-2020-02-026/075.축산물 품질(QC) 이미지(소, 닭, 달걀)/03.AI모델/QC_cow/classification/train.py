import sys
import os
import os.path as osp
import time
import copy
import argparse
import random
import shutil
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models



model_names= sorted(name for name in models.__dict__
                   if name.islower() and not name.startswith("__")
                    and callable(models.__dict__[name]))


parser=argparse.ArgumentParser(description='training pipeline for image classification')

parser.add_argument('--log_path', default='./log', type=str)  # 로그 텍스트를 저장할 위치
parser.add_argument('--gpu', default=0, help='gpu allocation')  # 사용할 GPU 선택

parser.add_argument('--model_name', default='resnext50_32x4d', choices=model_names)  # 사용할 모델 선택
parser.add_argument('--exp', type=str, help='model explanation', required=True)  # 훈련 방식 메모

parser.add_argument('--resize', default=512, type=int)  # 이미지 크기 재설정
parser.add_argument('--num_workers', default=6, type=int)  # 훈련에 사용할 CPU 코어 수

parser.add_argument('--epochs', default=200, type=int)  # 전체 훈련 epoch
parser.add_argument('--batch_size', default=8, type=int)  # 배치 사이즈
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float)  # learning rate
parser.add_argument('--momentum', default=0.9, type=float)  # optimizer의 momentum
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)  # 가중치 정규화


parser.add_argument('--optim', default='SGD')  # optimizer
parser.add_argument('--pretrained', dest='pretrained', type=str, help='use pre-trained model')  # pre-train 모델 사용 여부

args=parser.parse_args()



# make folders
if not os.path.exists('./log'):
    os.mkdir('./log')
    
if not os.path.exists('./model_weight'):
    os.mkdir('./model_weight')
    
if not os.path.exists('./checkpoint'):
    os.mkdir('./checkpoint')


def log(message):
    with open(osp.join(args.log_path, args.model_name)+'_'+f"{args.exp}"+'.txt', 'a+') as logger:
        logger.write(f'{message}\n')

    
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

def main_worker(args):    
    log(f'model name: {args.model_name}')
    log(f'Explanation: {args.exp}')
    log(f'num_workers: {args.num_workers}')
    log(f'n_epochs: {args.epochs}')
    log(f'batch_size: {args.batch_size}')
    log(f'Resize: {args.resize}')  
    

    
    # model arrangement   

    if args.pretrained:            
        log(f'\n=> using pre-trained model {args.model_name}')

    else:
        log(f'\n=> creating model {args.model_name}')
        model=models.__dict__[args.model_name](pretrained=False)     

    
    if 'res' in str(args.model_name):
        model.fc=nn.Linear(model.fc.in_features, 5)
        
    elif 'vgg' in str(args.model_name):
        model.classifier[-4]=nn.Linear(in_features=4096, out_features=256)
        model.classifier[-1]=nn.Linear(in_features=256, out_features=5)
        
    elif 'dense' in str(args.model_name):
        model.classifier=nn.Sequential(
            nn.Linear(model.classifier.in_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=5) )   
    
    model=model.to(device)
    
    criterion=nn.CrossEntropyLoss().to(device)
    
    if args.optim=='SGD':
        optimizer=optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        lr_scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

        
    log(f"optimizer: {optimizer}")
      
    
    # train transforms
    train_compose=transforms.Compose([
        transforms.Resize([args.resize, args.resize]),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(90),        
        transforms.ToTensor()
    ])
    log(train_compose)
    
    
    # validation transforms
    valid_compose=transforms.Compose([
        transforms.Resize([args.resize, args.resize]),
        transforms.ToTensor()
    ])    
    
    base_root='../Dataset/preprocessed/classification'
    train_dataset=datasets.ImageFolder(osp.join(base_root, 'train'), transform=train_compose)
    valid_dataset=datasets.ImageFolder(osp.join(base_root, 'val'), transform=valid_compose)
    test_dataset=datasets.ImageFolder(osp.join(base_root, 'test'), transform=valid_compose)
    
    log(f'train size : {len(train_dataset)}')
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
    args.start_epoch=0
    best_loss=np.inf
    best_val_acc=0.0
    best_val_f1=0.0
    early_stop_count=0
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
        test(test_loader, model, criterion, args)
        

        lr_scheduler.step(epoch_loss)
        current_lr=optimizer.state_dict()['param_groups'][0]['lr']

        if previous_lr > current_lr:
            log(f"\n+ Learning Rate was decreased from {previous_lr} to {current_lr} +")
            previous_lr=current_lr
            
        # remenber best and save checkpoint
        is_best_acc=best_val_acc<val_acc
        best_val_acc=max(best_val_acc, val_acc)   
        
        is_best_f1=best_val_f1<val_f1
        best_val_f1=max(best_val_f1, val_f1)
        
        is_best=best_loss>val_loss
        best_loss=min(best_loss, val_loss)
        
        save_checkpoint({
            'epoch': epoch+1,
            'arch': args.model_name,
            'state_dict': model.state_dict(),
            'best_val_loss': best_loss,
            'best_val_acc' : best_val_acc,
            'best_val_f1' : best_val_f1,
            'optimizer': optimizer.state_dict()
        }, is_best, is_best_acc, is_best_f1)
                
        if is_best:
            log('\n---- Best Val Loss ----')
            
        if is_best_acc:
            log('\n---- Best Val Accuracy ----')
            
        if is_best_f1:
            log('\n---- Best Val F1-Score')
            
        end=time.time()
        
        log(f'\nRunning Time: {int((end-since)//60)}m {int((end-since)%60)}s\n\n')
        
        # early stopping
        if is_best_acc:
            early_stop_count=0
        else:
            early_stop_count+=1
            
        if early_stop_count==20:
            log(f'\nEarly Stopped because Validation Acc is not increasing for 20 epochs')
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
    
    for i, (images, target) in enumerate(train_loader):                 
        
        images=images.to(device)
        target=target.to(device)         
              
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                output=model(images)
                loss=criterion(output, target)
        else:
            output=model(images)
            loss=criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()       
        optimizer.step()
        
        running_loss+=loss.item()*images.size(0)
        
        # accuracy
        
        _, output_index=torch.max(output, 1)
        
        total+=target.size(0)
        correct+=(output_index==target).sum().float()  
        
        tot_labels.extend(list(target.cpu().numpy()))
        tot_pred_labels.extend(list(output_index.view(-1).cpu().numpy()))  
    
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
            images=images.to(device)
            target=target.to(device)            
     
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    output=model(images)
                    loss=criterion(output, target)
            else:
                output=model(images)
                loss=criterion(output, target)
            
            running_loss+=loss.item()*target.size(0)
            
            _, output_index=torch.max(output, 1)
            
            total+=target.size(0)
            correct+=(output_index==target).sum().float()
            
            tot_labels.extend(list(target.cpu().numpy()))
            tot_pred_labels.extend(list(output_index.view(-1).cpu().numpy()))          
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
            images=images.to(device)
            target=target.to(device)
              
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    output=model(images)
                    loss=criterion(output, target)
            else:
                output=model(images)
                loss=criterion(output, target)
            
            running_loss+=loss.item()*target.size(0)
            
            _, output_index=torch.max(output, 1)
            
            total+=target.size(0)
            correct+=(output_index==target).sum().float()
            
            tot_labels.extend(list(target.cpu().numpy()))
            tot_pred_labels.extend(list(output_index.view(-1).cpu().numpy()))          
        acc=100*correct/total
        
        test_loss=running_loss / len(test_loader.dataset)
        
        log(f"[+] Test Accuracy: {acc :.3f},  Test Loss: {test_loss :.4f}")
        
        f1, re, pre=calculate_scores(tot_labels, tot_pred_labels)
        log(f"[+]  F1: {f1 :.3f},  Precision: {pre :.3f},  ReCall: {re :.3f}\n")


def calculate_scores(tot_labels, tot_pred_labels):
    f1=f1_score(tot_labels, tot_pred_labels, average='macro')
    re=recall_score(tot_labels, tot_pred_labels, average='macro')
    pre=precision_score(tot_labels, tot_pred_labels, average='macro', zero_division=0)
    
    return f1, re, pre




def save_checkpoint(state, is_best, is_best_acc, is_best_f1, filename='./checkpoint/'+args.model_name+'_'+args.exp+'.pht'):
    torch.save(state, filename)
    if is_best_acc:
        shutil.copyfile(filename, './model_weight/'+args.model_name+'_'+args.exp+'_best_Acc.pth')
    
    if is_best:
        shutil.copyfile(filename, './model_weight/'+args.model_name+'_'+args.exp+'_best_Loss.pth')
        
    if is_best_f1:
        shutil.copyfile(filename, './model_weight/'+args.model_name+'_'+args.exp+'_best_F1.pth')

        

    
if __name__=='__main__':
    main_worker(args)
    
    