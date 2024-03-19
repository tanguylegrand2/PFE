import numpy as np
import torch
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepMusicModel(nn.Module):
    def __init__(self, output_size):
        super(DeepMusicModel, self).__init__()
        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        
        # Define the second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        
        # Define the third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        
        # Define the fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        
        # Fully connected layer
        self.fc1 = nn.Linear(20736, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Softmax layer 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Apply the first convolutional layer and normalization, followed by ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Apply the second convolutional layer and normalization, followed by ReLU
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Apply the third convolutional layer and normalization, followed by ReLU
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Apply the fourth convolutional layer and normalization, followed by ReLU
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Reshape for the fully connected layer
        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        # Apply the dropout layer
        x = self.dropout(x)

        # Apply the softmax layer
        x = self.softmax(x)

        return x



    def forward(self, x):
        # Apply the first convolutional layer and normalization, followed by ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Apply the second convolutional layer and normalization, followed by ReLU
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Apply the third convolutional layer and normalization, followed by ReLU
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Apply the fourth convolutional layer and normalization, followed by ReLU
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Reshape for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Apply the fully connected layer
        x = self.fc1(x)
        
        # Apply the dropout layer
        x = self.dropout(x)

        # Apply the softmax layer
        x = self.softmax(x)

        return x





def save_models_hyperparams_and_metadata(models, hyperparameters_list, metadata_list, directory_name):
    # Base directory
    base_dir = 'Models'

    # Create a specific directory for the models within the base directory
    models_dir = os.path.join(base_dir, directory_name)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    for i, model in enumerate(models):
        # Get current date and time for filename
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Define file names for model, hyperparameters, and metadata
        model_filename = f"{directory_name}-{i}-{current_time}.pth"
        hyperparams_filename = f"{directory_name}-{i}-{current_time}_hyperparameters.json"
        metadata_filename = f"{directory_name}-{i}-{current_time}_metadata.json"
        model_path = os.path.join(models_dir, model_filename)
        hyperparams_path = os.path.join(models_dir, hyperparams_filename)
        metadata_path = os.path.join(models_dir, metadata_filename)

        # Save the model's state dictionary
        torch.save(model.state_dict(), model_path)

        # Save the hyperparameters in JSON format
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparameters_list[i], f, indent=4)

        # Save the training metadata in JSON format
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list[i], f, indent=4)

def _save_single_model(model, hyperparameters, training_metadata, model_dir, directory_name, index=None):
    # Get current date and time for filename
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    index_suffix = f"-{index}" if index is not None else ""

    # Define paths for model, hyperparameters, and metadata
    model_filename = f"{directory_name}{index_suffix}-{current_time}.pth"
    hyperparams_filename = f"{directory_name}{index_suffix}-{current_time}_hyperparameters.json"
    metadata_filename = f"{directory_name}{index_suffix}-{current_time}_metadata.json"
    model_path = os.path.join(model_dir, model_filename)
    hyperparams_path = os.path.join(model_dir, hyperparams_filename)
    metadata_path = os.path.join(model_dir, metadata_filename)

    # Save the model's state dictionary
    torch.save(model.state_dict(), model_path)

    # Save the hyperparameters in JSON format
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    # Save the training metadata in JSON format
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=4)


def load_model_or_models(directory_name, model_class):
    base_dir = 'Models'
    model_dir = os.path.join(base_dir, directory_name)

    # Trouver les fichiers de modèle et de métadonnées dans le dossier
    files = os.listdir(model_dir)
    model_files = [f for f in files if f.endswith('.pth')]
    hyperparams_files = [f for f in files if f.endswith('_hyperparameters.json')]

    models = []
    hyperparameters_list = []

    # Charger les modèles et hyperparamètres
    for model_file, hyperparams_file, output in zip(model_files, hyperparams_files):
        model_path = os.path.join(model_dir, model_file)
        hyperparams_path = os.path.join(model_dir, hyperparams_file)

        # Charger l'état du modèle
        model = model_class()  # Créer une instance de la classe du modèle
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Mettre le modèle en mode évaluation
        models.append(model)

        # Charger les hyperparamètres
        with open(hyperparams_path, 'r') as f:
            hyperparameters = json.load(f)
        hyperparameters_list.append(hyperparameters)

    # Si un seul modèle est chargé, retourner le modèle seul, sinon retourner la liste
    if len(models) == 1:
        return models[0], hyperparameters_list[0]
    else:
        return models, hyperparameters_list
    
def load_model_without_hyperparams(directory_name, model_class):
    base_dir = 'Models'
    model_dir = os.path.join(base_dir, directory_name)
    
    # Trouver le fichier de modèle et de métadonnées dans le dossier
    files = os.listdir(model_dir)
    model_file = [f for f in files if f.endswith('.pth')][0]
    
    model_path = os.path.join(model_dir, model_file)

    # Charger l'état du modèle
    model = model_class()  # Créer une instance de la classe du modèle
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Mettre le modèle en mode évaluation

    return model

def load_deep_music(directory_name, model_class, outputs):
    base_dir = 'Models'
    model_dir = os.path.join(base_dir, directory_name)

    # Trouver les fichiers de modèle et de métadonnées dans le dossier
    files = os.listdir(model_dir)
    model_files = [f for f in files if f.endswith('.pth')]
    hyperparams_files = [f for f in files if f.endswith('_hyperparameters.json')]

    models = []
    hyperparameters_list = []

    # Charger les modèles et hyperparamètres
    for model_file, hyperparams_file, output in zip(model_files, hyperparams_files, outputs):
        model_path = os.path.join(model_dir, model_file)
        hyperparams_path = os.path.join(model_dir, hyperparams_file)

        # Charger l'état du modèle
        model = model_class(output)  # Créer une instance de la classe du modèle
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Mettre le modèle en mode évaluation
        models.append(model)

        # Charger les hyperparamètres
        with open(hyperparams_path, 'r') as f:
            hyperparameters = json.load(f)
        hyperparameters_list.append(hyperparameters)

    # Si un seul modèle est chargé, retourner le modèle seul, sinon retourner la liste
    if len(models) == 1:
        print(models[0], hyperparameters_list[0])
        return models[0], hyperparameters_list[0]
    else:
        return models, hyperparameters_list




def plot_mse(model1_results,model2_results):
    # Création des graphiques
    plt.figure(figsize=(10, 5))

    # Graphique 1 : Comparaison des MSE pour chaque SNR
    plt.subplot(1, 2, 1)
    plt.plot(model1_results['SNR'], model1_results['MSE'], marker='o', label='Modèle 1')
    plt.plot(model2_results['SNR'], model2_results['MSE'], marker='o', label='Modèle 2')
    plt.xlabel('SNR (dB)')
    plt.ylabel('MSE (degrés)')
    plt.title('Comparaison des MSE pour chaque SNR')
    plt.legend()

    # Graphique 2 : Comparaison des performances des modèles à différents SNR
    plt.subplot(1, 2, 2)
    plt.bar([1, 2], [model1_results['MSE'][0], model2_results['MSE'][0]], tick_label=['Modèle 1', 'Modèle 2'], label='-10 dB')
    plt.bar([1, 2], [model1_results['MSE'][1], model2_results['MSE'][1]], tick_label=['Modèle 1', 'Modèle 2'], label='0 dB', alpha=0.5)
    plt.xlabel('Modèle')
    plt.ylabel('MSE (degrés)')
    plt.title('Comparaison des performances des modèles à différents SNR')
    plt.legend()

    plt.tight_layout()
    plt.show()


