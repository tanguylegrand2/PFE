import numpy as np
import torch
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt


def save_model_hyperparams_and_metadata(model, hyperparameters, training_metadata, directory_name):
    # Base directory
    base_dir = 'Models'
    
    # Create a specific directory for the model within the base directory
    model_dir = os.path.join(base_dir, directory_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Get current date and time for filename
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Define paths for model, hyperparameters, and metadata
    model_filename = f"{directory_name}-{current_time}.pth"
    hyperparams_filename = f"{directory_name}-{current_time}_hyperparameters.json"
    metadata_filename = f"{directory_name}-{current_time}_metadata.json"
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


def load_model_and_hyperparams(directory_name, model_class):
    base_dir = 'Models'
    model_dir = os.path.join(base_dir, directory_name)
    
    # Trouver le fichier de modèle et de métadonnées dans le dossier
    files = os.listdir(model_dir)
    model_file = [f for f in files if f.endswith('.pth')][0]
    hyperparams_file = [f for f in files if f.endswith('_hyperparameters.json')][0]
    
    model_path = os.path.join(model_dir, model_file)
    hyperparams_path = os.path.join(model_dir, hyperparams_file)

    # Charger l'état du modèle
    model = model_class()  # Créer une instance de la classe du modèle
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Mettre le modèle en mode évaluation

    # Charger les hyperparamètres
    with open(hyperparams_path, 'r') as f:
        hyperparameters = json.load(f)

    return model, hyperparameters



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


