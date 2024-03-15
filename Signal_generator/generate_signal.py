import numpy as np

def generate_noise(sd, size):
    # Générer un bruit aléatoire complexe avec une distribution gaussienne
    real_part = np.random.normal(0, sd, size)
    imag_part = np.random.normal(0, sd, size)
    # Créer un tableau complexe à partir des parties réelle et imaginaire
    noise_vector = np.vectorize(complex)(real_part, imag_part)
    return noise_vector

def generate_steering_vector(nbSensors, theta, perturbation_parameter_sd=0, d=1, wavelength=2, get_CramerRao_data=False):
    # Génération du steering vector
    steering_vector = []
    if get_CramerRao_data:
        D_t = []
        for i in range(nbSensors):
            # Calcul du perturbation_parameter
            perturbation_parameter = np.random.normal(1, perturbation_parameter_sd)
            steering_vector.append(np.exp(-1j * perturbation_parameter * (i+1) * 2 * np.pi * d / wavelength * np.sin(np.radians(theta))))
            D_t.append(-1j * perturbation_parameter * (i+1) * 2 * np.pi * d / wavelength * np.cos(np.radians(theta)) * np.exp(-1j * perturbation_parameter * (i+1) * 2 * np.pi * d / wavelength * np.sin(np.radians(theta))))
        return steering_vector, D_t
    else:
        for i in range(nbSensors):
            # Calcul du perturbation_parameter
            perturbation_parameter = np.random.normal(1, perturbation_parameter_sd)
            steering_vector.append(np.exp(-1j * perturbation_parameter * (i+1) * 2 * np.pi * d / wavelength * np.sin(np.radians(theta))))
        return steering_vector

def generate_S_matrix(nbSources, nbTimePoints, var_ratio, correlation_List, SNR_dB, get_CramerRao_data=False):
    # Conversion du SNR en linéaire
    SNR_linear = 10 ** (SNR_dB / 10)
    
    # Initialisation des variances des signaux basées sur le rapport de variance et le SNR
    # On commence par une liste de variances hypothétiques où le premier signal a une variance de 1
    variances_hypo = [1] + [1 * ratio for ratio in var_ratio]
    
    # Ajustement des variances pour respecter le SNR global
    total_variances_hypo = sum(variances_hypo)
    var1 = SNR_linear / total_variances_hypo  # Ajustement de la variance du premier signal pour respecter le SNR
    adjusted_variances = [var1] + [var1 * ratio for ratio in var_ratio]
    
    # Gestion des erreurs pour la corrélation et les variances
    if len(correlation_List) != nbSources * (nbSources - 1) // 2:
        raise ValueError("Incorrect number of correlation coefficients.")
    
    # Construction de la matrice de covariance avec les variances ajustées et les corrélations
    covariance_matrix = np.zeros((nbSources, nbSources))
    k = 0
    for i in range(nbSources):
        for j in range(i + 1, nbSources):
            covariance_matrix[i, j] = _get_covariance(correlation_List[k], adjusted_variances[i], adjusted_variances[j])
            k += 1
    covariance_matrix += covariance_matrix.T
    covariance_matrix += np.diag(adjusted_variances)
    
    # Génération de la matrice S avec la décomposition de Cholesky
    L_cholesky = np.linalg.cholesky(covariance_matrix)
    S = np.dot(np.random.normal(0, 1, (nbTimePoints, nbSources)) + 1j * np.random.normal(0, 1, (nbTimePoints, nbSources)), L_cholesky.T)
    if get_CramerRao_data:
        return S, covariance_matrix
    else:
        return S

def _get_covariance(correlation_coefficient, varA, varB):
    # Calcule une covariance à partir d'une corrélation
    covariance = correlation_coefficient * np.sqrt(varA * varB) # Calcul de la covariance
    return covariance

def generate_A_matrix(nbSensors, thetaList, perturbation_parameter_sd, get_CramerRao_data=False): # Génération de la Steering matrix
    A = []
    if get_CramerRao_data:
        D = []
        for theta in thetaList:
            A_t, D_t = generate_steering_vector(nbSensors, theta, perturbation_parameter_sd, get_CramerRao_data=True)
            A.append(A_t)
            D.append(D_t)
        A = np.transpose(np.array(A))
        D = np.transpose(np.array(D))
        return A, D
    else:
        for theta in thetaList:
            A_t = generate_steering_vector(nbSensors, theta, perturbation_parameter_sd)
            A.append(A_t)
        A = np.transpose(np.array(A))
        return A

def generate_X_matrix(nbSources, nbSensors, nbTimePoints, thetaList, var_ratio, correlation_List, signal_noise_ratio, perturbation_parameter_sd, get_CramerRao_data=False): # Génération de la matrice des signaux reçus
    # Vérification que le nombre d'angles theta fournis est bon
    if len(thetaList) != nbSources:
        raise ValueError("Provide the correct number of thetas. You need to have one theta for each source.")
    # Création des matrices S et A
    if get_CramerRao_data:
        S, S_covariance_matrix = generate_S_matrix(nbSources, nbTimePoints, var_ratio, correlation_List, signal_noise_ratio, get_CramerRao_data)
        A, D = generate_A_matrix(nbSensors, thetaList, perturbation_parameter_sd, get_CramerRao_data)
    else:
        S = generate_S_matrix(nbSources, nbTimePoints, var_ratio, correlation_List, signal_noise_ratio)
        A = generate_A_matrix(nbSensors, thetaList, perturbation_parameter_sd)

    # Initialisation de X
    X = []

    # Implémentation du rapport signal sur bruit (Signal Noise Ratio, SNR)
    noise_power = 1
    #Création de la matrice X
    for i in range(nbTimePoints):
        b_t = generate_noise(np.sqrt(noise_power), nbSensors) # Bruit
        X_t = A @ S[i] + b_t # Signal reçu à un instant t
        X.append(X_t)
    X = np.array(X)

    if get_CramerRao_data:
        return X, A, S_covariance_matrix, D
    else:
        return X

def generate_R_hat(X):
    # Calcul de la matrice de covariance R_hat du signal reçu
    R_hat = np.cov(X, rowvar=False)
    return R_hat

def generate_R_hat_with_phase(X):
    """
    Calcul de la matrice de covariance R_hat du signal reçu avec un canal supplémentaire pour la phase.
    
    Args:
    - X (np.ndarray): La matrice des signaux reçus de forme (nbTimePoints, nbSensors), où chaque élément est complexe.
    
    Returns:
    - R_hat_extended (np.ndarray): Une version étendue de la matrice de covariance avec un canal supplémentaire pour la phase.
      Sa forme sera (3, nbSensors, nbSensors), où les deux premiers canaux sont les parties réelle et imaginaire de R_hat,
      et le troisième canal est la phase de chaque élément.
    """
    R_hat = np.cov(X, rowvar=False)
    phase = np.angle(R_hat)
    
    # Séparez les parties réelle et imaginaire de R_hat
    real_part = np.real(R_hat)
    imag_part = np.imag(R_hat)
    
    # Empilez les trois canaux
    R_hat_extended = np.stack((real_part, imag_part, phase), axis=-1)
    R_hat_extended = np.transpose(R_hat_extended, (2, 0, 1))
    
    return R_hat_extended


def generate_R_hat_with_phase_complex(X):
    """
    Calcul de la matrice de covariance R_hat du signal reçu avec un canal supplémentaire pour la phase en restant en complexe.
    
    Args:
    - X (np.ndarray): La matrice des signaux reçus de forme (nbTimePoints, nbSensors), où chaque élément est complexe.
    
    Returns:
    - R_hat_extended (np.ndarray): Une version étendue de la matrice de covariance avec un canal supplémentaire pour la phase.
      Sa forme sera (2, nbSensors, nbSensors), où le premier canal est la covariance de R_hat,
      et le deuxième canal est la phase de chaque élément.
    """
    R_hat = np.cov(X, rowvar=False)
    phase = np.angle(R_hat)
    
    # Empilez les deux canaux
    R_hat_extended = np.stack((R_hat, phase), axis=-1)
    R_hat_extended = np.transpose(R_hat_extended, (2, 0, 1))
    
    return R_hat_extended