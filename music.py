import numpy as np
from generate_signal import generate_steering_vector, generate_R_hat
from draw_plot import plot_single_music
from scipy.signal import find_peaks

def music_method(X, nbSensors, nbSources, print_angles = False, draw_plot = False):
    angles_range = np.linspace(-90, 90, 1801) # Angles que l'on souhaite tester
    R_hat = generate_R_hat(X)

    # Calculer les vecteurs propres et les valeurs propres
    _, eigenvectors = np.linalg.eigh(R_hat)
    # Sélectionner les k plus grandes valeurs propres
    noise_subspace = eigenvectors[:, :nbSources]

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
    all_peaks, _ = find_peaks(np.real(music_spectrum), height=0)  # height=0 pour inclure tous les pics
    sorted_peaks = sorted(all_peaks, key=lambda x: music_spectrum[x], reverse=True) # Triez les pics par amplitude dans l'ordre décroissant
    top_peaks = sorted(sorted_peaks[:nbSources]) # Sélectionnez les deux plus grands pics
    estimated_angles = angles_range[top_peaks] # Obtenez les angles estimés correspondant aux deux pics
    return estimated_angles