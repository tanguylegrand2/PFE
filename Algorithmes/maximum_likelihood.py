import numpy as np
from scipy.optimize import minimize
from Signal_generator.generate_signal import generate_steering_vector, generate_R_hat

def maximum_likelihood_estimation(X, nbSensors, nbSources, print_angles=False):
    initial_guess = np.linspace(-45, 45, nbSources)

    def objective_function(angles):
        R_hat = generate_R_hat(X)
        likelihood = 0
        for angle in angles:
            a = generate_steering_vector(nbSensors, angle).reshape(-1, 1)
            likelihood += np.real(np.log(np.linalg.det(a @ a.conj().T + R_hat)))
        return -likelihood

    # Utilisez 'initial_guess' dans l'appel à minimize
    result = minimize(objective_function, initial_guess, bounds=[(-90, 90)] * nbSources, method='L-BFGS-B')

    if result.success:
        estimated_angles = result.x
    else:
        estimated_angles = None  # Gestion d'erreur

    if print_angles:
        print("Angles estimés:", estimated_angles)

    return estimated_angles
