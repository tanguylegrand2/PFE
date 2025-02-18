# PFE

# Signal Arrival Direction Estimation: Classical Techniques vs Deep Learning Models

This project focuses on the comparison of classical estimation techniques and deep learning models for **Direction of Arrival (DOA)** estimation. The goal is to assess the performance of these different approaches in the context of signal processing.

## Project Objective

The objective of this project is to explore, implement, and compare classical signal estimation methods (such as MUSIC and ESPRIT) with deep learning models (such as CNNs and RNNs) for the task of DOA estimation.

## Technologies and Tools

- **Programming Language**: Python
- **Deep Learning**: Keras, TensorFlow (if applicable for deep learning models)
- **Signal Processing**: Classical methods for DOA estimation
- **Data**: Simulated signal data for model training and evaluation
- **Plotting and Visualization**: Matplotlib, Seaborn

## Key Features

- **Classical Estimation Methods**: Implementation of classical algorithms like MUSIC and ESPRIT for DOA estimation.
- **Deep Learning Models**: Implementation of neural network-based models (such as CNNs and RNNs) for comparison.
- **Performance Comparison**: Metrics and visualizations comparing the accuracy and robustness of classical vs deep learning methods.

## Installation

Clone the repository and install the required libraries:

```bash
git clone https://github.com/tanguylegrand2/PFE.git
cd PFE
```

You can install the necessary dependencies using `requirements.txt` or manually:

```bash
pip install numpy scipy matplotlib tensorflow
```
## Running the Project

To run the project, simply execute the relevant Jupyter notebooks or Python scripts for each method. The following files demonstrate the different techniques:

- **`DAO_Estimator.ipynb`**: Implementation of classical estimation methods.
- **`DeepMusic.ipynb`**: Jupyter notebook showcasing deep learning-based DOA estimation.
- **`MSE_compare.ipynb`**: Notebook for comparing MSE between classical and deep learning models.

## Main Files

- **`Signal_generator.py`**: Contains code for generating synthetic signal data for testing.
- **`comparison_scripts/`**: Scripts for comparing the performance of different methods.
- **`Models/`**: Contains implementations of deep learning models for DOA estimation.
- **`Plots/`**: Plotting scripts for visualizing results.

## Results

The project provides insights into how classical methods and deep learning techniques perform in terms of estimation accuracy, robustness, and computational complexity.

## Future Improvements

- **Model Optimization**: Improve deep learning models for better performance.
- **Real-World Testing**: Apply these models to real-world data for further evaluation.
- **Hybrid Approaches**: Investigate hybrid models combining classical and deep learning techniques.


