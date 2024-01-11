import numpy as np
from generate_signal import generate_steering_vector, generate_R_hat
from draw_plot import plot_single_beaforming
from scipy.signal import find_peaks

def beamforming_method(X, nbSensors, nbSources, print_angles = False, draw_plot = False):
    angles_range = np.linspace(-90, 90, 1801) # Angles que l'on souhaite tester
    R_hat = generate_R_hat(X)
    inv_R_hat = np.linalg.pinv(R_hat)  # Pseudo-inverse de la matrice de covariance
    beamformer_output = np.zeros_like(angles_range, dtype=complex)

    for i, angle in enumerate(angles_range):
        # Calcul du vecteur de poids pour chaque direction
        steering_vector_angle = generate_steering_vector(nbSensors, angle)
        w = np.dot(inv_R_hat, steering_vector_angle) / np.dot(np.conj(steering_vector_angle), np.dot(inv_R_hat, steering_vector_angle))
        # Calcul de la sortie du beamformer pour cette direction
        beamformer_output[i] = np.abs(np.dot(np.conj(w), np.dot(R_hat, w)))

    # Normalisation pour une meilleure visualisation
    beamformer_output /= np.max(beamformer_output)
    # Estimation des angles
    estimated_angles = estimate_angles(nbSources, beamformer_output, angles_range)

    if print_angles: # print les angles estimés si True
        print(estimated_angles)
    if draw_plot: # Trace un graphique du spectre de beamforming si True
        plot_single_beaforming(beamformer_output, angles_range)

    return estimated_angles

def estimate_angles(nbSources, beamformer_output, angles_range):
    all_peaks, _ = find_peaks(np.real(beamformer_output), height=0)  # height=0 pour inclure tous les pics
    sorted_peaks = sorted(all_peaks, key=lambda x: beamformer_output[x], reverse=True) # Triez les pics par amplitude dans l'ordre décroissant
    top_peaks = sorted(sorted_peaks[:nbSources]) # Sélectionnez les deux plus grands pics
    estimated_angles = angles_range[top_peaks] # Obtenez les angles estimés correspondant aux deux pics
    return estimated_angles