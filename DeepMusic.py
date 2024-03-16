from Signal_generator.generate_signal import generate_X_matrix
from Algorithmes.beamforming import beamforming_method
from Signal_generator.generate_signal import generate_A_matrix
from Signal_generator.generate_signal import generate_S_matrix
from Signal_generator.generate_signal import generate_noise
from Signal_generator.generate_signal import generate_R_hat
from Signal_generator.generate_signal import generate_R_hat_with_phase
from Algorithmes.music import music_method
from Algorithmes.music import estimate_angles
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import itertools
from torchsummary import summary
from numpy.linalg import eigh
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import math
import random
from sklearn.metrics import mean_squared_error
from Algorithmes.music import generate_steering_vector
from Plots.draw_plot import plot_single_music
from Models.compare import *
from torch.utils.data import Dataset, DataLoader
import time as time



def calculate_music_spectrum(R_hat, nbSensors, nbSources, angles_range):
    # Calculer les vecteurs propres et les valeurs propres
    _, eigenvectors = np.linalg.eigh(R_hat)
    # Sélectionner les k plus grandes valeurs propres
    noise_subspace = eigenvectors[:, :-nbSources]
    music_spectrum = np.zeros_like(angles_range, dtype=float)
    for idx, theta in enumerate(angles_range):
        steering_vector = generate_steering_vector(nbSensors, theta)
        music_spectrum[idx] = 1 / np.linalg.norm(noise_subspace.conj().T @ steering_vector)

    return music_spectrum

def generate_combinations(phi_max, rho, nb_sources):
    """Cette fonction sert à générer toutes les combinaisons d'angles possibles dans la plage de résolution"""
    # Générer une plage de valeurs possibles pour les signaux
    values = list(range(-int(phi_max / rho), int(phi_max / rho) + 1))
    
    # Générer toutes les combinaisons possibles de signaux
    all_combinations = list(itertools.product(values, repeat=nb_sources))
    
    # Supprimer les combinaisons où l'ordre ne compte pas et où deux signaux ont la même valeur
    unique_combinations = {tuple(sorted(combination)) for combination in all_combinations if len(set(combination)) == nb_sources}
    
    return list(unique_combinations)


def generate_deepmusic_partitioned_data(nbSensors, nbSources, T, SNR_TRAIN, Q, N, phi_max, rho, correlation_List, var_ratio):
    angles_combination = generate_combinations(phi_max, rho, nbSources)
    training_data_sets = [[] for _ in range(Q)]
    L = N // Q  # Assuming N is divisible by Q

    i = 0
    for signal_noise_ratio in SNR_TRAIN:
        for angles in angles_combination:
            thetaList = list(angles)

            X = generate_X_matrix(nbSources=nbSources, nbSensors=nbSensors, nbTimePoints=nbTimePoints, thetaList=thetaList, var_ratio=var_ratio, correlation_List=correlation_List, signal_noise_ratio=signal_noise_ratio, perturbation_parameter_sd=perturbation_parameter_sd)
            R_hat_with_phase = generate_R_hat_with_phase(X)
            R_hat = generate_R_hat(X)

            full_theta = np.linspace(- phi_max , phi_max, N)
            full_music_spectrum = calculate_music_spectrum(R_hat=R_hat,nbSensors=nbSensors, nbSources=nbSources, angles_range=full_theta)

            for q in range(Q):
                # Sampling MUSIC spectrum for the q-th subregion
                start_index = q * L
                end_index = start_index + L
                pq = full_music_spectrum[start_index:end_index]
                
                input_data = R_hat_with_phase
                output_data = pq
                training_data_sets[q].append((input_data, output_data))
            i += 1

    return training_data_sets


class DeepMusicDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data, output_data = self.data[idx]
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32)

def create_loaders_for_subregion(data, batch_size=128, validation_split=0.2, test_split=0.2):
    # Séparation des données en ensembles d'entraînement, de validation et de test
    train_data, test_data = train_test_split(data, test_size=test_split, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=validation_split, random_state=42)

    # Création des instances de DeepMusicDataset
    train_dataset = DeepMusicDataset(train_data)
    val_dataset = DeepMusicDataset(val_data)
    test_dataset = DeepMusicDataset(test_data)

    # Création des DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



class DeepMusicModel(nn.Module):
    def __init__(self):
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
        
        # Assuming the spatial dimensions (height and width) are reduced to 1x1 after the convolutions
        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=20736, out_features=N//Q)
        
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
        
        # Flatten the tensor for the fully connected layer
        x = torch.flatten(x, 1)
        
        # Apply the first fully connected layer
        x = self.fc1(x)
        
        # Apply the dropout layer
        x = self.dropout(x)

        #Apply the softmax layer
        x = self.softmax(x)
        
        return x



def deep_music_pred(X,nbSensors,nbSources,models):
    a = [model(torch.tensor(generate_R_hat_with_phase(X), dtype=torch.float32).unsqueeze(0)) for model in models[0]]
    a = torch.cat(a, dim=0).detach().numpy()
    return estimate_angles(nbSources,a, np.linspace(- phi_max, phi_max,N))

def calculate_accuracy(predicted_spectrum, target_spectrum, angles_range, nbSources, peak_tolerance):
    """
    Calculate accuracy for a single predicted and target spectrum.
    
    :param predicted_spectrum: Predicted MUSIC spectrum (1D NumPy array).
    :param target_spectrum: True MUSIC spectrum (1D NumPy array).
    :param angles_range: The range of angles over which the MUSIC spectrum is calculated.
    :param nbSources: The number of sources (peaks) to consider.
    :param peak_tolerance: The tolerance for considering a predicted peak to match a target peak.
    :return: Accuracy as a float.
    """
    # Estimate angles from the predicted and target spectra
    estimated_angles = estimate_angles(nbSources, predicted_spectrum, angles_range)
    true_angles = estimate_angles(nbSources, target_spectrum, angles_range)

    # Compare the estimated angles with the true angles
    correct_predictions = sum(1 for est_angle in estimated_angles if np.any(np.abs(true_angles - est_angle) <= peak_tolerance))
    accuracy = correct_predictions / nbSources if nbSources > 0 else 0
    return accuracy



def train_and_evaluate_spectrum(loaders_per_subregion, epochs, nbSources, angles_range, peak_tolerance=5):
    models = []
    train_loss_list = []
    val_loss_list = []
    val_mse_list = []

    # Training configuration
    initial_lr = 0.01
    lr_decay_factor = 0.5
    lr_decay_epoch = 10
    early_stopping_patience = 3

    start_time = time.time()  # Start time for training

    for train_loader, val_loader, test_loader in loaders_per_subregion:
        model = DeepMusicModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
        criterion = torch.nn.MSELoss()

        best_val_mse = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            # Adjust learning rate
            if epoch % lr_decay_epoch == 0 and epoch > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay_factor

            # Training loop
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets).item()
                optimizer.step()
                train_loss += loss
                
            avg_train_loss = train_loss / len(train_loader)
            train_loss_list.append(avg_train_loss)

            # Validation loop
            model.eval()
            val_loss, val_mse = 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    val_mse += F.mse_loss(outputs, targets).item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_mse = val_mse / len(val_loader)
            val_loss_list.append(avg_val_loss)
            val_mse_list.append(avg_val_mse)

            # Early stopping
            if avg_val_mse < best_val_mse:
                best_val_mse = avg_val_mse
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        models.append(model)

    # Calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")

    # Test set evaluation
    concatenated_outputs = []
    concatenated_targets = []

    for model, test_loader in zip(models, [x[2] for x in loaders_per_subregion]):
        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)

                # Concatenate outputs and targets
                concatenated_outputs.append(outputs)
                concatenated_targets.append(targets)

    # Calculate the average MSE per epoch on the validation set
    avg_mse_per_epoch_val = []
    for epoch in range(epochs):
        epoch_mse = []
        for i, model in enumerate(models):
            # Determine the actual number of epochs the model was trained for
            actual_epochs = len(train_loss_list[i * epochs:(i + 1) * epochs])
            if epoch < actual_epochs:
                epoch_mse.append(train_loss_list[i * epochs + epoch])
        avg_mse_per_epoch_val.append(np.mean(epoch_mse))

    print("Average MSE per epoch on validation set:", avg_mse_per_epoch_val)

    # Calculate the average MSE per epoch on the training set
    avg_mse_per_epoch_train = []
    for epoch in range(epochs):
        epoch_mse = []
        for i, model in enumerate(models):
            # Determine the actual number of epochs the model was trained for
            actual_epochs = len(val_mse_list[i * epochs:(i + 1) * epochs])
            if epoch < actual_epochs:
                epoch_mse.append(val_mse_list[i * epochs + epoch])
        avg_mse_per_epoch_train.append(np.mean(epoch_mse))

    print("Average MSE per epoch on training set:", avg_mse_per_epoch_train)

    # Concatenate outputs and targets over all batches
    concatenated_outputs = torch.cat(concatenated_outputs, dim=0)
    concatenated_targets = torch.cat(concatenated_targets, dim=0)

    # Compute MSE for concatenated outputs and targets
    total_mse = F.mse_loss(concatenated_outputs, concatenated_targets).item()

    print(f'Average Training Loss: {np.mean(train_loss_list):.4f}')
    print(f'Average Validation Loss: {np.mean(val_loss_list):.4f}')
    print(f'MSE on Test Set: {total_mse:.4f}')
    print(f'Training time: {training_time}')
    return models, {"MSE" : total_mse}

Q = 2
N = 121
phi_max = 60
rho = 1
nbSources = 2 # Nombre de sources
nbSensors = 9 # Nombre de capteurs
nbTimePoints = 100 # Nombre de points temporels
signal_noise_ratio = 10 # Rapport signal sur bruit en décibels. Si 'False', cela revient à une absence totale de bruit.
theta1 = -20 # Angle entre la perpendiculaire à la ligne de capteurs, et la source 1
theta2 = 20 # Angle entre la perpendiculaire à la ligne de capteurs, et la source 2
var_ratio = [1] # Liste qui donne le rapport entre la variance du signal 1 et celui des autres sources (ex: [2, 3] signifie que la source 2 a une variance 2 fois plus grande que la source 1, et la source 3 a une variance 3 fois plus grande que la source 1)
correlation_List = [0] # Liste des corrélations. Il y a une corrélation nécéssaire pour chaque paire distincte de sources différentes: 0 pour 1 source, 1 pour 2 sources, 3 pour 3 sources, 6 pour 4 sources etc...
# Ordre de remplisage de la correlation_List: de gauche à droite et ligne par ligne, moitié haut-droite de la matrice uniquement, puis symétrie de ces valeurs pour la moitié bas-gauche
perturbation_parameter_sd = 0 # Écart-type de la distribution normale qui génère les erreurs de calibration des capteurs

full_theta = np.linspace(- phi_max , phi_max, N)
# Example usage
SNR_TRAIN = [-20,-15,-10, -5, 0] # Different SNR levels for training
training_datasets = generate_deepmusic_partitioned_data(nbSensors=nbSensors, nbSources=nbSources, T=nbTimePoints, SNR_TRAIN=SNR_TRAIN, Q=Q, N=N, phi_max=phi_max, rho=rho, correlation_List=correlation_List, var_ratio=var_ratio) 
# Création des DataLoader pour chaque sous-région
loaders_per_subregion = [create_loaders_for_subregion(training_datasets[q]) for q in range(Q)]

from Models.compare import *
model1_results = train_and_evaluate_spectrum(loaders_per_subregion=loaders_per_subregion, epochs=1000, nbSources=nbSources, angles_range=full_theta, peak_tolerance=5)
models, metrics = model1_results
save_models_hyperparams_and_metadata(models, hyperparameters_list=[{"A" : "a"} for _ in range(Q)], metadata_list=[{"A" : "a"} for _ in range(Q)], directory_name="DeepMusic_test_4")