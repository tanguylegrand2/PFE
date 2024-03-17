from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import math
import itertools
from Signal_generator.generate_signal import generate_R_hat_with_phase
from Models.compare import *
phi_max =60  # φmax
rho = 1 # Résolution



#Définition du modèle
nombre_de_classe=phi_max*2/rho+1
nombre_de_classe=int(nombre_de_classe)
class DOACNN(nn.Module):
    def __init__(self, num_channels=3, num_classes=nombre_de_classe):
        super(DOACNN, self).__init__()

        # Couches convolutionnelles 2D
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1)

        # Normalisation de taille 256
        self.norm2 = nn.BatchNorm2d(256)
        self.norm5 = nn.BatchNorm2d(256)
        self.norm8 = nn.BatchNorm2d(256)
        self.norm11 = nn.BatchNorm2d(256)

        self.relu3 =  nn.ReLU()
        self.relu6 =  nn.ReLU()
        self.relu9 =  nn.ReLU()
        self.relu12 =  nn.ReLU()
        self.relu15 =  nn.ReLU()
        self.relu18 =  nn.ReLU()
        self.relu21 =  nn.ReLU()

        # Couches Dropout
        self.dropout16 = nn.Dropout(0.2)
        self.dropout19 = nn.Dropout(0.2)
        self.dropout22 = nn.Dropout(0.2)

        # Couche d'aplatissement
        self.flatten13 = nn.Flatten()

        # Couches entièrement connectées (FC)
        self.fc14 = nn.Linear(in_features=256 , out_features=4096)
        self.fc17 = nn.Linear(in_features=4096, out_features=2048)
        self.fc20 = nn.Linear(in_features=2048, out_features=1024)
        self.fc23 = nn.Linear(in_features=1024, out_features=num_classes)

        # Couche de sortie Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.norm5(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.norm8(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.norm11(x)
        x = self.relu12(x)
        x = self.flatten13(x)
        x = self.fc14(x)
        x = self.relu15(x)
        x = self.dropout16(x)
        x = self.fc17(x)
        x = self.relu18(x)
        x = self.dropout19(x)
        x = self.fc20(x)
        x = self.relu21(x)
        x = self.dropout22(x)
        x = self.fc23(x)
        x = self.sigmoid(x)

        return x
    
class DOACNN4096(nn.Module):
    def __init__(self, num_channels=3, num_classes=nombre_de_classe):
        super(DOACNN4096, self).__init__()

        # Couches convolutionnelles 2D
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1)

        # Normalisation de taille 256
        self.norm2 = nn.BatchNorm2d(256)
        self.norm5 = nn.BatchNorm2d(256)
        self.norm8 = nn.BatchNorm2d(256)
        self.norm11 = nn.BatchNorm2d(256)

        self.relu3 =  nn.ReLU()
        self.relu6 =  nn.ReLU()
        self.relu9 =  nn.ReLU()
        self.relu12 =  nn.ReLU()
        self.relu15 =  nn.ReLU()
        self.relu18 =  nn.ReLU()
        self.relu21 =  nn.ReLU()

        # Couches Dropout
        self.dropout16 = nn.Dropout(0.2)
        self.dropout19 = nn.Dropout(0.2)
        self.dropout22 = nn.Dropout(0.2)

        # Couche d'aplatissement
        self.flatten13 = nn.Flatten()

        # Couches entièrement connectées (FC)
        self.fc14 = nn.Linear(in_features=4096, out_features=4096)
        self.fc17 = nn.Linear(in_features=4096, out_features=2048)
        self.fc20 = nn.Linear(in_features=2048, out_features=1024)
        self.fc23 = nn.Linear(in_features=1024, out_features=num_classes)

        # Couche de sortie Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.norm5(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.norm8(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.norm11(x)
        x = self.relu12(x)
        x = self.flatten13(x)
        x = self.fc14(x)
        x = self.relu15(x)
        x = self.dropout16(x)
        x = self.fc17(x)
        x = self.relu18(x)
        x = self.dropout19(x)
        x = self.fc20(x)
        x = self.relu21(x)
        x = self.dropout22(x)
        x = self.fc23(x)
        x = self.sigmoid(x)

        return x
    
class DOACNN6400(nn.Module):
    def __init__(self, num_channels=3, num_classes=nombre_de_classe):
        super(DOACNN6400, self).__init__()

        # Couches convolutionnelles 2D
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1)

        # Normalisation de taille 256
        self.norm2 = nn.BatchNorm2d(256)
        self.norm5 = nn.BatchNorm2d(256)
        self.norm8 = nn.BatchNorm2d(256)
        self.norm11 = nn.BatchNorm2d(256)

        self.relu3 =  nn.ReLU()
        self.relu6 =  nn.ReLU()
        self.relu9 =  nn.ReLU()
        self.relu12 =  nn.ReLU()
        self.relu15 =  nn.ReLU()
        self.relu18 =  nn.ReLU()
        self.relu21 =  nn.ReLU()

        # Couches Dropout
        self.dropout16 = nn.Dropout(0.2)
        self.dropout19 = nn.Dropout(0.2)
        self.dropout22 = nn.Dropout(0.2)

        # Couche d'aplatissement
        self.flatten13 = nn.Flatten()

        # Couches entièrement connectées (FC)
        self.fc14 = nn.Linear(in_features=6400, out_features=4096)
        self.fc17 = nn.Linear(in_features=4096, out_features=2048)
        self.fc20 = nn.Linear(in_features=2048, out_features=1024)
        self.fc23 = nn.Linear(in_features=1024, out_features=num_classes)

        # Couche de sortie Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.norm5(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.norm8(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.norm11(x)
        x = self.relu12(x)
        x = self.flatten13(x)
        x = self.fc14(x)
        x = self.relu15(x)
        x = self.dropout16(x)
        x = self.fc17(x)
        x = self.relu18(x)
        x = self.dropout19(x)
        x = self.fc20(x)
        x = self.relu21(x)
        x = self.dropout22(x)
        x = self.fc23(x)
        x = self.sigmoid(x)

        return x

# Création d'une instance du modèle
model = DOACNN()




from Models.compare import load_model_without_hyperparams

loaded_DAO_9_500_30=load_model_without_hyperparams("DAO_9_500_30", model_class=DOACNN)

def DAO_9_500_30(X,nbSensors, nbSources):
    R_hat_with_phase = generate_R_hat_with_phase(X)
    X_train = torch.from_numpy(R_hat_with_phase).to(dtype=torch.float32)
    X_train=X_train.unsqueeze(0)
    pred=loaded_DAO_9_500_30(X_train)
    _, angles_predit = pred.topk(nbSources, dim=1)
    sorted_angles, _ = torch.sort(angles_predit)
    sorted_angles=sorted_angles.tolist()[0]
    for e in range(len(sorted_angles)):
        sorted_angles[e]=sorted_angles[e]-60
    return np.array(sorted_angles)



loaded_DAO_16_500_20=load_model_without_hyperparams("DAO_16_500_20", model_class=DOACNN4096)

def DAO_16_500_20(X,nbSensors, nbSources):
    R_hat_with_phase = generate_R_hat_with_phase(X)
    X_train = torch.from_numpy(R_hat_with_phase).to(dtype=torch.float32)
    X_train=X_train.unsqueeze(0)
    pred=loaded_DAO_16_500_20(X_train)
    _, angles_predit = pred.topk(nbSources, dim=1)
    sorted_angles, _ = torch.sort(angles_predit)
    sorted_angles=sorted_angles.tolist()[0]
    for e in range(len(sorted_angles)):
        sorted_angles[e]=sorted_angles[e]-60
    return np.array(sorted_angles)


loaded_DAO_16_500_20_padding=load_model_without_hyperparams("DAO_16_500_20_padding", model_class=DOACNN6400)

def DAO_16_500_20_padding(X,nbSensors, nbSources):
    R_hat_with_phase = generate_R_hat_with_phase(X)
    X_train = torch.from_numpy(R_hat_with_phase).to(dtype=torch.float32)
    X_train=X_train.unsqueeze(0)
    pred=loaded_DAO_16_500_20_padding(X_train)
    _, angles_predit = pred.topk(nbSources, dim=1)
    sorted_angles, _ = torch.sort(angles_predit)
    sorted_angles=sorted_angles.tolist()[0]
    for e in range(len(sorted_angles)):
        sorted_angles[e]=sorted_angles[e]-60
    return np.array(sorted_angles)

loaded_DAO_16_500_200=load_model_without_hyperparams("DAO_16_500_200", model_class=DOACNN6400)

def DAO_16_500_200(X,nbSensors, nbSources):
    R_hat_with_phase = generate_R_hat_with_phase(X)
    X_train = torch.from_numpy(R_hat_with_phase).to(dtype=torch.float32)
    X_train=X_train.unsqueeze(0)
    pred=loaded_DAO_16_500_200(X_train)
    _, angles_predit = pred.topk(nbSources, dim=1)
    sorted_angles, _ = torch.sort(angles_predit)
    sorted_angles=sorted_angles.tolist()[0]
    for e in range(len(sorted_angles)):
        sorted_angles[e]=sorted_angles[e]-60
    return np.array(sorted_angles)