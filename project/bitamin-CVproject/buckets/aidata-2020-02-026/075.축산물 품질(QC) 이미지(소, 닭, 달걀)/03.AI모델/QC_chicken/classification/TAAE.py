import torch
import torch.nn as nn

from torchvision import models

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        nn.init.xavier_uniform(m.bias)
        
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
        
#Encoder
class Encoder(nn.Module):  
    def __init__(self, base_model="VGG19", dropout_p=0.2):
        super(Encoder, self).__init__()
        
        if base_model == "VGG19":
            self.model_ft = models.vgg19(pretrained=True)
            for param in self.model_ft.parameters():
                param.requires_grad = True
            self.model_ft.classifier = nn.Sequential(*list(self.model_ft.classifier.children())[:-6])
            self.X_dim = self.model_ft.classifier[0].out_features
        elif base_model == "RESNET50":
            self.model_ft = models.resnet50(pretrained=False)
            for param in self.model_ft.parameters():
                param.requires_grad = True
            self.X_dim = self.model_ft.fc.in_features
            self.model_ft.fc = Identity()
        elif base_model == "RESNET101":
            self.model_ft = models.resnet101(pretrained=True)
            for param in self.model_ft.parameters():
                param.requires_grad = True
            self.X_dim = self.model_ft.fc.in_features
            self.model_ft.fc = Identity()
        elif base_model == "RESNET152":
            self.model_ft = models.resnet152(pretrained=True)
            for param in self.model_ft.parameters():
                param.requires_grad = True
            self.X_dim = self.model_ft.fc.in_features
            self.model_ft.fc = Identity()
        elif base_model == "DENSENET161":
            self.model_ft = models.densenet161(pretrained=True)
            for param in self.model_ft.parameters():
                param.requires_grad = True
            self.X_dim = self.model_ft.classifier.in_features
            self.model_ft.classifier = Identity()
        elif base_model == "DENSENET201":
            self.model_ft = models.densenet201(pretrained=True)
            for param in self.model_ft.parameters():
                param.requires_grad = True
            self.X_dim = self.model_ft.classifier.in_features
            self.model_ft.classifier = Identity()
        
        assert self.X_dim # check whether base model properly constructed
        self.dropout_p = dropout_p
        
        # Fully_connected Layers
        self.encoder_fc = nn.Sequential()

        # X_dim to X_dim/4
        self.encoder_fc.add_module("encoder_fc1", nn.Linear(int(self.X_dim), int(self.X_dim/4)))
        self.encoder_fc.add_module("encoder_bn1", nn.BatchNorm1d(int(self.X_dim/4)))
        self.encoder_fc.add_module("encoder_activation1", nn.ReLU(inplace=True))
        self.encoder_fc.add_module("encoder_dropout1", nn.Dropout(self.dropout_p))
        # X_dim/4 to X_dim/16                           
        self.encoder_fc.add_module("encoder_fc2", nn.Linear(int(self.X_dim/4), int(self.X_dim/16)))

        # X_dim/8 to X_dim/16       
        init_weights(self.encoder_fc)
        self.z_dim = int(self.X_dim/16)  # z_dim=256

    def forward(self, images):
        x = self.model_ft(images)
        z = self.encoder_fc(x)
        return x, z
    
    def get_z_dim(self):
        return self.z_dim
    
#Decoder
class Decoder(nn.Module):  
    def __init__(self, z_dim, dropout_p=0.2):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        
        # Fully_connected Layers
        self.decoder_fc = nn.Sequential()

        # z_dim to z_dim*4
        self.decoder_fc.add_module("decoder_fc1", nn.Linear(int(self.z_dim), int(self.z_dim*4)))
        self.decoder_fc.add_module("decoder_bn1", nn.BatchNorm1d(int(self.z_dim*4)))
        self.decoder_fc.add_module("decoder_activation1", nn.ReLU(inplace=True))
        self.decoder_fc.add_module("decoder_dropout1", nn.Dropout(self.dropout_p))
        
        # z_dim*4 to z_dim*8
        self.decoder_fc.add_module("decoder_fc2", nn.Linear(int(self.z_dim*4), int(self.z_dim*8)))
        self.decoder_fc.add_module("decoder_bn2", nn.BatchNorm1d(int(self.z_dim*8)))
        self.decoder_fc.add_module("decoder_activation2", nn.ReLU(inplace=True))
        self.decoder_fc.add_module("decoder_dropout2", nn.Dropout(self.dropout_p))

        # z_dim*8 to z_dim*16
        self.decoder_fc.add_module("decoder_fc3", nn.Linear(int(self.z_dim*8), int(self.z_dim*16)))

        init_weights(self.decoder_fc)

    def forward(self, z):
        x_recon = self.decoder_fc(z)
        return x_recon

# Discriminator
class Discriminator(nn.Module):  
    def __init__(self, z_dim, dropout_p=0.2, n_target=1):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        self.n_target = n_target
        
        # Fully_connected Layers
        self.discriminator_fc = nn.Sequential()
        # z_dim to z_dim/4
        self.discriminator_fc.add_module("discriminator_fc1", nn.Linear(int(self.z_dim), int(self.z_dim/4)))
        self.discriminator_fc.add_module("discriminator_bn1", nn.BatchNorm1d(int(self.z_dim/4)))
        self.discriminator_fc.add_module("discriminator_activation1", nn.ReLU(inplace=True))
        self.discriminator_fc.add_module("discriminator_dropout1", nn.Dropout(self.dropout_p))

        # z_dim/4 to n_target
        self.discriminator_fc.add_module("discriminator_fc2", nn.Linear(int(self.z_dim/4), int(self.n_target)))
        if self.n_target == 1:
            self.discriminator_fc.add_module("discriminator_activation4", nn.Sigmoid())
        else:
            self.discriminator_fc.add_module("discriminator_activation4", nn.Softmax(dim=1))
        init_weights(self.discriminator_fc)
        
    def forward(self, z):
        output = self.discriminator_fc(z)
        return output