import numpy as np

def generate_noise(sd, size):
    # Générer un bruit aléatoire complexe avec une distribution gaussienne
    real_part = np.random.normal(0, sd, size)
    imag_part = np.random.normal(0, sd, size)
    # Créer un tableau complexe à partir des parties réelle et imaginaire
    noise_vector = np.vectorize(complex)(real_part, imag_part)
    return noise_vector

def generate_steering_vector(nbSensors, theta, d = 1, wavelength = 2):
    # Génération du steering vector
    steering_vector = np.exp(-1j * np.arange(nbSensors) * 2 * np.pi * d / wavelength * np.sin(np.radians(theta)))
    return steering_vector

def generate_S_matrix(nbSources, nbTimePoints, varList, correlation_List):
    # Vérification que le nombre de variances fournies est bon
    if len(varList) != nbSources:
        raise ValueError("Provide the correct number of variances. You need to have one variance for each source.")
    # Vérification que le nombre de corrélations fournies est bon
    if len(correlation_List) != nbSources * (nbSources - 1) // 2:
        raise ValueError("Provide the correct number of correlation coefficients. You need to have one correlation for each distinct pair of sources.")

    # Création d'une matrice carrée vide
    covariance_matrix = np.zeros((nbSources, nbSources))
    # Remplissage de la moitié haut-droite de la matrice de covariance
    k = 0
    for i in range(nbSources):
        for j in range(i + 1, nbSources):
            covariance_matrix[i, j] = _get_covariance(correlation_List[k], varList[i], varList[j])
            k+=1
    # Remplissage par symétrie de la moitié bas-droite
    covariance_matrix += covariance_matrix.T
    # Remplissage de la diagonale de la marice de covariance
    covariance_matrix += np.diag(varList)
    print(covariance_matrix)
    # Utiliser la décomposition de Cholesky pour obtenir une matrice L telle que L * transpose(L) = covariance_matrix
    L_cholesky = np.linalg.cholesky(covariance_matrix)
    # Générer la matrice S en utilisant la matrice L_cholesky
    S = np.dot(np.random.normal(0, 1, (nbTimePoints, nbSources)) + 1j * np.random.normal(0, 1, (nbTimePoints, nbSources)), L_cholesky.T)  # Chaque ligne est un signal source
    return S

def _get_covariance(correlation_coefficient, varA, varB):
    # Calcule une covariance à partir d'une corrélation
    covariance = correlation_coefficient * np.sqrt(varA * varB) # Calcul de la covariance
    return covariance

def generate_A_matrix(nbSensors, thetaList): # Génération de la Steering matrix
    A = []
    for theta in thetaList:
        A_t = generate_steering_vector(nbSensors, theta)
        A.append(A_t)
    A = np.transpose(np.array(A))
    return A

def generate_X_matrix(nbSources, nbSensors, nbTimePoints, thetaList, varList, correlation_List, signal_noise_ratio): # Génération de la matrice des signaux reçus
    # Vérification que le nombre d'angles theta fournis est bon
    if len(thetaList) != nbSources:
        raise ValueError("Provide the correct number of thetas. You need to have one theta for each source.")
    # Création dela matrice S
    S = generate_S_matrix(nbSources, nbTimePoints, varList, correlation_List)
    # Création dela matrice A
    A = generate_A_matrix(nbSensors, thetaList)
    # Initialisation de X
    X = []

    # Implémentation du rapport signal sur bruit (Signal Noise Ratio, SNR)
    if signal_noise_ratio is not False:
        signal_power = np.mean(np.abs(S)**2) # Calcul de la puissance du signal à partir de la matrice S
        noise_power = signal_power / 10**(signal_noise_ratio / 10) # Calculez la puissance du bruit en fonction du SNR
        #Création de la matrice X
        for i in range(nbTimePoints):
            b_t = generate_noise(np.sqrt(noise_power), nbSensors) # Bruit
            X_t = np.dot(A, S[i]) + b_t # Signal reçu à un instant t
            X.append(X_t)
        X = np.array(X)
    else:
        #Création de la matrice X
        for i in range(nbTimePoints):
            b_t = generate_noise(np.sqrt(2**-30), nbSensors) # Le Beamforming ne fonctionne pas si je ne rajoute pas un bruit minime pour une raison qui m'échappe
            X_t = np.dot(A, S[i]) + b_t# Signal reçu à un instant t
            X.append(X_t)
            if i < 1:
                print(np.dot(A, S[i]))
                print(X_t)
        X = np.array(X)
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
      Sa forme sera (nbSensors, nbSensors, 3), où les deux premiers canaux sont les parties réelle et imaginaire de R_hat,
      et le troisième canal est la phase de chaque élément.
    """
    R_hat = np.cov(X, rowvar=False)
    phase = np.angle(R_hat)
    
    # Séparez les parties réelle et imaginaire de R_hat
    real_part = np.real(R_hat)
    imag_part = np.imag(R_hat)
    
    # Empilez les trois canaux
    R_hat_extended = np.stack((real_part, imag_part, phase), axis=-1)
    
    return R_hat_extended