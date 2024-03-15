import numpy as np
from Signal_generator.generate_signal import generate_steering_vector, generate_R_hat
from Plots.draw_plot import plot_single_music
from scipy.signal import find_peaks

def music_method(X, nbSensors, nbSources, print_angles = False, draw_plot = False):
    angles_range = np.linspace(-90, 90, 1801) # Angles que l'on souhaite tester
    R_hat = generate_R_hat(X)

    # Calculer les vecteurs propres et les valeurs propres
    _, eigenvectors = np.linalg.eigh(R_hat)
    # Sélectionner les k plus grandes valeurs propres
    noise_subspace = eigenvectors[:, :-nbSources]

    # Calcul du spectre MUSIC
    music_spectrum = np.zeros_like(angles_range, dtype=float)
    for idx, theta in enumerate(angles_range):
        steering_vector = generate_steering_vector(nbSensors, theta)
        music_spectrum[idx] = 1 / np.linalg.norm(noise_subspace.conj().T @ steering_vector)
    
    
    # Estimation des angles
    estimated_angles = estimate_angles(nbSources, music_spectrum, angles_range)

    if print_angles: # print les angles estimés si True
        print(estimated_angles)
    if draw_plot: # Trace un graphique du spectre de beamforming si True
        plot_single_music(music_spectrum, angles_range)

    return estimated_angles

def estimate_angles(nbSources, music_spectrum, angles_range):
    # Ensure that music_spectrum is a 1-D array
    if not np.ndim(music_spectrum) == 1:
        # Flatten the array if it's multidimensional
        music_spectrum = music_spectrum.flatten()

    all_peaks, _ = find_peaks(np.real(music_spectrum), height=0)
    sorted_peaks = sorted(all_peaks, key=lambda x: music_spectrum[x], reverse=True)
    top_peaks = sorted(sorted_peaks[:nbSources])
    estimated_angles = angles_range[top_peaks]
    return estimated_angles


