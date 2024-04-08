import os
import numpy as np
import pandas as pd
import math
import random

from sklearn.metrics import f1_score

import torch
from torch.autograd import Variable

import sys
from TAAE import Encoder, Decoder, Discriminator

import explain_train


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad=data_grad.sign()
    perturbed_image=image+epsilon*sign_data_grad
    perturbed_image=torch.clamp(perturbed_image, 0, 1)
    return perturbed_image   

# Create functions    
def Create_Models(base_model="RESNET50", dropout_p=0.2, is_classifier=False, n_classes=0, device=None):
    if is_classifier and n_classes < 1:
        print("Please set n_classes more than 1 for classifier!!")
        assert n_classes > 1
    
    models = {}
    
    # Encoder
    models["Encoder"] = Encoder(base_model=base_model, dropout_p=dropout_p)
    # Decoder
    z_dim = models["Encoder"].get_z_dim()
    models["Decoder"] = Decoder(z_dim=z_dim, dropout_p=dropout_p)
    # Discriminator
    models["Discriminator"] = Discriminator(z_dim=z_dim, dropout_p=dropout_p, n_target=1)
    if is_classifier and n_classes > 1:
        models["Classifier"] = Discriminator(z_dim=z_dim, dropout_p=dropout_p, n_target=n_classes)
        
    if device :
        for key in models.keys():
            models[key].to(device)
            
    models["Z_dim"] = z_dim
    
    return models


# Train
def Train(train_loader, models, criterions, optimizers, device=None, n_trial_DC=5, log=None, args=None):
    EPS = 1e-15
    
    total_recon_loss = []
    total_disc_loss = []
    total_gen_loss = []
    total_cls_loss = []
    
    total_FGSM_recon_loss = []  
    
    #for accuracy
    total=0
    correct=0
    tot_labels=[]
    tot_pred_labels=[]
    
    # Set train mode
    for key in list(models.keys()):
        if key != "Z_dim":
            models[key].train()

    # train loader
    for i, (images, target) in enumerate(train_loader):    
        
        if device :
            real_data_v = Variable(images).to(device)
        else:
            real_data_v = Variable(images)
            
        """ Reconstruction loss """
        models["Encoder"].zero_grad()
        models["Decoder"].zero_grad()
        models["Discriminator"].zero_grad()
        for p in models["Discriminator"].parameters():
            p.requires_grad = False
        
        with torch.cuda.amp.autocast():
            real_data_v.requires_grad=True
            x, z = models["Encoder"](real_data_v)
            x_recon = models["Decoder"](z)
            reconstruction_loss = criterions["Reconstruction"](x_recon[target!=1], x[target!=1]).mean()
            
            if sum(target==1)!=0:
                loss_invert= (args.ld)/(criterions["Reconstruction"](x_recon[target==1], x[target==1]).mean() + 1e-5) 
                
                reconstruction_loss+=(loss_invert)
            
        total_recon_loss.append(reconstruction_loss.item()+1e-6)
        optimizers["Encoder"].zero_grad()
        optimizers["Decoder"].zero_grad()
        reconstruction_loss.backward()    
        optimizers["Encoder"].step()           
        optimizers["Decoder"].step()  
               
        
        
        #---------------------------------------
        # call FGSM Attack
        if sum(target==0)>1:
            recon_loss=0.0

            if random.random()<0.1:
                with torch.cuda.amp.autocast():
                    data_grad=real_data_v.grad.data[target==0]
                    normal=real_data_v[target==0]
                    new_anomaly=torch.empty_like(normal)                  

                    for i, (data, grad_) in enumerate(zip(normal, data_grad), 0):
                        perturbed=fgsm_attack(data, 0.15, grad_)
                        new_anomaly[i]=perturbed
                    
                    x, z=models['Encoder'](new_anomaly)
                    x_recon=models['Decoder'](z)
                    recon_loss += 1/(criterions['Reconstruction'](x_recon, x).mean() + 1e-5)

                    total_FGSM_recon_loss.append(recon_loss.item()+1e-5)
                    optimizers["Encoder"].zero_grad()
                    optimizers["Decoder"].zero_grad()
                    recon_loss.backward()    
                    optimizers["Encoder"].step()           
                    optimizers["Decoder"].step()        
         #---------------------------------------  
        
        
        
        
        
        """ Discriminator loss """       
        for p in models["Discriminator"].parameters():  
            p.requires_grad = True 
        models["Encoder"].eval()
        temp_disc_loss = []
        for _ in range(n_trial_DC):  
            models["Encoder"].zero_grad()
            models["Discriminator"].zero_grad()
            if device :
                real_data_v = Variable(images).to(device)
                real_gauss = Variable(torch.randn(images.size()[0], models["Z_dim"])).to(device)
            else:
                real_data_v = Variable(images)
                real_gauss = Variable(torch.randn(images.size()[0], models["Z_dim"]))
            
            with torch.cuda.amp.autocast():
                D_real_gauss = models["Discriminator"](real_gauss)

                _, fake_gauss = models["Encoder"](real_data_v)
                D_fake_gauss = models["Discriminator"](fake_gauss)

                D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))
            temp_disc_loss.append(D_loss.item())
            
            optimizers["Discriminator"].zero_grad()
            D_loss.backward()
            optimizers["Discriminator"].step()    
        total_disc_loss.append(np.mean(temp_disc_loss))
        
        """ Generator (Encoder) """
        models["Encoder"].zero_grad()
        models["Discriminator"].zero_grad()
        
        models["Encoder"].train()
        if device :
            real_data_v = Variable(images).to(device)
        else:
            real_data_v = Variable(images)
        
        with torch.cuda.amp.autocast():
            _, fake_gauss = models["Encoder"](real_data_v)
            D_fake_gauss = models["Discriminator"](fake_gauss)

            G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))
            total_gen_loss.append(G_loss.item())
        
        optimizers["Encoder_reg"].zero_grad()
        G_loss.backward()
        optimizers["Encoder_reg"].step()  

    
    log(f"[+] Train Recon Loss: {np.mean(total_recon_loss) :.3f},  FGSM_Recon Loss: {np.mean(total_FGSM_recon_loss) :.3f},  Disc Loss: {np.mean(total_disc_loss) :.3},  Gen Loss: {np.mean(total_gen_loss) :.3f}")
            
    return np.mean(total_recon_loss), np.mean(total_disc_loss), np.mean(total_gen_loss)



# Validation
def Validation(valid_loader, models, criterions, device=None, log=None, args=None):
    recon_loss = []
    cls_loss = []
    
    total=0
    correct=0
    
    tot_labels=[]
    tot_pred_labels=[]
    
    # Set eval mode
    for key in list(models.keys()):
        if key != "Z_dim":
            models[key].eval()
            
    # valid loader
    
    recon_dict={'recon_error':[], 'true_label_tot': tot_labels}
    
    
    for i, (images, target) in enumerate(valid_loader): 
        target=target[target!=99]
        images=images[~images.isnan()].view(-1, 3, 448, 448)  
        
        recon_dict['true_label_tot'].extend(list(target.numpy()))
        
        
        if device:
            real_data_v = Variable(images).to(device)
            target = Variable(target).to(device)
        else:
            real_data_v = Variable(images)
            target = Variable(target)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                x, z = models["Encoder"](real_data_v)
                x_recon = models["Decoder"](z)
                reconstruction_loss = criterions["Reconstruction"](x_recon, x).mean(dim=1)
                #loss dict for accuracy
                recon_dict['recon_error'].extend(reconstruction_loss.detach().cpu().numpy().tolist())
                
                # loss for export
                reconstruction_loss = criterions["Reconstruction"](x_recon[target!=1], x[target!=1]).mean()
                if sum(target==1)!=0:
                    loss_invert= (args.ld)/(criterions["Reconstruction"](x_recon[target==1], x[target==1]).mean() + 1e-5)     
                    reconstruction_loss+=(loss_invert)  

                recon_loss.append(reconstruction_loss.mean().item()+1e-6)
                
                
                
    result=explain_train.Prediction(recon_dict['recon_error'], recon_dict['true_label_tot'], test=False)
    _,_, f1, best_f1_threshold=result.get_prediction()
    
               
    #acc=100*correct/total    
    log(f"[+] Val Recon Loss: {np.mean(recon_loss) :.4f},  Val Tot F1: {f1 :.3f} with threshold={best_f1_threshold :.4f}")
    
    return np.mean(recon_loss), f1, best_f1_threshold

# Test
def Test(test_loader, models, criterions, device=None, log=None, args=None, best_f1_threshold=None):
    recon_loss = []
    cls_loss = []
    
    total=0
    correct=0
    
    tot_labels=[]
    tot_pred_labels=[]
    
    # Set eval mode
    for key in list(models.keys()):
        if key != "Z_dim":
            models[key].eval()
            
    # valid loader
    
    recon_dict={'recon_error':[], 'true_label_tot': tot_labels}
    
    
    for i, (images, target) in enumerate(test_loader): 
        target=target[target!=99]
        images=images[~images.isnan()].view(-1, 3, 448, 448)  
        
        recon_dict['true_label_tot'].extend(list(target.numpy()))
        
        
        if device:
            real_data_v = Variable(images).to(device)
            target = Variable(target).to(device)
        else:
            real_data_v = Variable(images)
            target = Variable(target)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                x, z = models["Encoder"](real_data_v)
                x_recon = models["Decoder"](z)
                reconstruction_loss = criterions["Reconstruction"](x_recon, x).mean(dim=1)
                #loss dict for accuracy
                recon_dict['recon_error'].extend(reconstruction_loss.detach().cpu().numpy().tolist())
                
                # loss for export
                reconstruction_loss = criterions["Reconstruction"](x_recon[target!=1], x[target!=1]).mean()
                if sum(target==1)!=0:
                    loss_invert= (args.ld)/(criterions["Reconstruction"](x_recon[target==1], x[target==1]).mean() + 1e-5)     
                    reconstruction_loss+=(loss_invert)  

                recon_loss.append(reconstruction_loss.mean().item()+1e-6)
                
    if best_f1_threshold:
                
        result=explain_train.Prediction(recon_dict['recon_error'], recon_dict['true_label_tot'], test=True, best_f1_threshold=best_f1_threshold)
        
        f1, best_f1_threshold=result.get_prediction()


        #acc=100*correct/total    
        log(f"[+] Test Recon Loss: {np.mean(recon_loss) :.4f},  Test Tot F1: {f1 :.3f} with threshold={best_f1_threshold :.4f}")