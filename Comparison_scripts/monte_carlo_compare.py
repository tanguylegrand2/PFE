import numpy as np
import matplotlib.pyplot as plt
import time
from Signal_generator.generate_signal import generate_X_matrix

def run_comparison(parameter_to_compare, algorithms_to_compare, nbiterations, nbSources, nbSensors, theta, nbTimePoints, snr, correlation, var_ratio, perturbation_parameter_sd, two_symetrical_angles=True):
    valid_parameters = ["snr", "nbTimePoints", "correlation", "var_ratio", "perturbation_parameter_sd"]
    if parameter_to_compare not in valid_parameters:
        raise ValueError(f"parameter_to_compare doit être l'un des suivants : {valid_parameters}")

    # Mapping du paramètre à comparer avec ses valeurs
    if parameter_to_compare == "snr":
        parameter_to_compare_values = snr
    elif parameter_to_compare == "nbTimePoints":
        parameter_to_compare_values = nbTimePoints
    elif parameter_to_compare == "correlation":
        parameter_to_compare_values = correlation
    elif parameter_to_compare == "var_ratio":
        parameter_to_compare_values = var_ratio
    elif parameter_to_compare == "perturbation_parameter_sd":
        parameter_to_compare_values = perturbation_parameter_sd
    
    # Préparation des tableaux pour stocker les EQM pour chaque algorithme
    MSE_results = {name: np.zeros(len(parameter_to_compare_values)) for name in algorithms_to_compare.keys()}
    Cramer_Rao = np.zeros(len(parameter_to_compare_values))

    execution_times = {name: [] for name in algorithms_to_compare.keys()}

    # Boucle principale pour les iterations
    for i, value in enumerate(parameter_to_compare_values):
        algorithm_estimations = {name: [] for name in algorithms_to_compare.keys()}
        original_counts = {name: 0 for name in algorithms_to_compare.keys()}
        all_Cramer_Rao = []

        # Iterations
        for _ in range(nbiterations):
            # Génération du signal
            if parameter_to_compare == "snr":
                X, A, P, D = generate_X_matrix(nbSources, nbSensors, nbTimePoints, theta, var_ratio, correlation, value, perturbation_parameter_sd, get_CramerRao_data=True)
            elif parameter_to_compare == "nbTimePoints":
                X, A, P, D = generate_X_matrix(nbSources, nbSensors, value, theta, var_ratio, correlation, snr, perturbation_parameter_sd, get_CramerRao_data=True)
            elif parameter_to_compare == "correlation":
                X, A, P, D = generate_X_matrix(nbSources, nbSensors, nbTimePoints, theta, var_ratio, value, snr, perturbation_parameter_sd, get_CramerRao_data=True)
            elif parameter_to_compare == "var_ratio":
                X, A, P, D = generate_X_matrix(nbSources, nbSensors, nbTimePoints, theta, value, correlation, snr, perturbation_parameter_sd, get_CramerRao_data=True)
            elif parameter_to_compare == "perturbation_parameter_sd":
                X, A, P, D = generate_X_matrix(nbSources, nbSensors, nbTimePoints, theta, var_ratio, correlation, snr, value, get_CramerRao_data=True)
            
            # Exécution de chaque algorithme
            for name, algorithm_function in algorithms_to_compare.items():
                start_time = time.time()  # Début de mesure du temps
                estimated_angles = algorithm_function(X, nbSensors, nbSources)
                end_time = time.time()  # Fin de mesure du temps

                algorithm_estimations[name].append(estimated_angles)
                original_counts[name] += 1
                execution_times[name].append(end_time - start_time)  # Ajoute le temps d'exécution à la liste

            if parameter_to_compare != "nbTimePoints":
                single_Cramer_Rao = get_CramerRao(nbTimePoints, A, P, D)
            else:
                single_Cramer_Rao = get_CramerRao(value, A, P, D)
            all_Cramer_Rao.append(single_Cramer_Rao)

        print(f"Pour {parameter_to_compare} = {value} :")

        # Calcul de l'EQM pour chaque algorithme
        for name in algorithms_to_compare.keys():
            print("-----")
            clean_estimation = remove_outliers([theta] * nbiterations, algorithm_estimations[name])
            removed_outliers = original_counts[name] - len(clean_estimation[0])
            percent_outliers = 100 * removed_outliers / original_counts[name]
            average_execution_time = np.mean(execution_times[name])
            print(f"Temps moyen d'estimation pour une itération de {name}: {average_execution_time:.4f} secondes")
            print(f"{name}: {removed_outliers} outliers removed ({percent_outliers:.2f}%)")
            if len(clean_estimation[0]) > 0:
                MSE_results[name][i] = calculate_MSE(clean_estimation[0], clean_estimation[1], two_symetrical_angles)
                print(f"Valeur pour {name} : {MSE_results[name][i]}")
            else:
                MSE_results[name][i] = np.nan

        # Calcul de la borne de Cramer-Rao pour la comparaison
        Cramer_Rao[i] = np.mean(all_Cramer_Rao)
        print(f"-----\nValeur de la Cramer Rao Lower Bound : {Cramer_Rao[i]}")
        print("--------------------------------")

    # Affichage des résultats
    plt.figure(figsize=(10, 6))
    for name, mse in MSE_results.items():
        plt.plot(parameter_to_compare_values, mse, label=name)
    plt.plot(parameter_to_compare_values, Cramer_Rao, label='Cramer-Rao Bound', linestyle='--', marker='^', color='red')
    plt.title(f"MSE en fonction de la variable {parameter_to_compare}")
    plt.xlabel(parameter_to_compare)
    plt.ylabel("Root Mean Square Error (MSE)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()


# Fonction pour supprimer les outliers et retourner les listes nettoyées
def remove_outliers(real_theta_list, theta_hat_list):
    real_theta_clean, theta_hat_clean = [], []
    for real_theta, theta_hat in zip(real_theta_list, theta_hat_list):
        if all(abs(rt - th) <= 5 for rt, th in zip(real_theta, theta_hat)):  # On retire l'outlier si l'estimation d'un des deux angles est à plus de 4 degrés de la valeur réelle
            real_theta_clean.append(real_theta)
            theta_hat_clean.append(theta_hat)
    return real_theta_clean, theta_hat_clean

# Fonction pour calculer l'EQM
def calculate_MSE(real_theta, theta_hat, two_symetrical_angles):
    if two_symetrical_angles:
        mse = np.mean([(rt - th) ** 2 for rt, th in zip(real_theta, theta_hat)])
    else:
        mse = np.mean([(rt[0] - th[0]) ** 2 for rt, th in zip(real_theta, theta_hat)])
    return mse

def get_CramerRao(nbTimePoints, A, P, D):
    noise_variance = 1  # Sigma est fixé à 1
    term = np.linalg.inv(((D.conj().T @ (np.eye(A.shape[0]) - A @ np.linalg.inv(A.conj().T @ A) @ A.conj().T) @ D) * P.T).real)
    crlb = np.diag(term)[0] / (2 * nbTimePoints) * noise_variance
    #print(f"-----\nValeur de la Cramer Rao Lower Bound : {crlb}")
    return crlb*7000