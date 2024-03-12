import numpy as np
import torch 
import os


def save_model_and_metadata(model, directory_name):
    # Base directory
    base_dir = 'Models'
    
    # Create a specific directory for the model within the base directory
    model_dir = os.path.join(base_dir, directory_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define paths for model and metadata
    model_path = os.path.join(model_dir, '{}.pth'.format(directory_name))
    metadata_path = os.path.join(model_dir, '{}_metadata.txt'.format(directory_name))

    # Save the model's state dictionary
    torch.save(model.state_dict(), model_path)

    # Save the model's architecture
    with open(metadata_path, 'w') as f:
        f.write(str(model))
