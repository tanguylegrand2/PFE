import numpy as np
import matplotlib.pyplot as plt

def plot_single_beaforming(beamformer_output, angles_range): # Visualisation d'un spectre Beamforming
    plt.figure(figsize=(10, 6))
    plt.plot(angles_range, np.real(beamformer_output))
    plt.title('Beamforming Spatial')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Normalized signal power')
    plt.grid(True)
    plt.show()

def plot_single_music(music_spectrum, angles_range): # Visualisation d'un spectre MUSIC
    plt.figure(figsize=(10, 6))
    plt.plot(angles_range, np.real(music_spectrum))
    plt.title('MUSIC spectrum')
    plt.xlabel('Angle (degrés)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()