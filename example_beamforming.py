from Signal_generator.generate_signal import generate_X_matrix
from Algorithmes.beamforming import beamforming_method

nbSources = 2 # Nombre de sources
nbSensors = 9 # Nombre de capteurs
nbTimePoints = 100 # Nombre de points temporels
signal_noise_ratio = 3 # Rapport signal sur bruit en décibels. Si 'False', cela revient à une absence totale de bruit.
theta1 = -7 # Angle entre la perpendiculaire à la ligne de capteurs, et la source 1
theta2 = 7 # Angle entre la perpendiculaire à la ligne de capteurs, et la source 2
var_ratio = [5] # Liste qui donne le rapport entre la variance du signal 1 et celui des autres sources (ex: [2, 3] signifie que la source 2 a une variance 2 fois plus grande que la source 1, et la source 3 a une variance 3 fois plus grande que la source 1)
correlation_List = [0.4] # Liste des corrélations. Il y a une corrélation nécéssaire pour chaque paire distincte de sources différentes: 0 pour 1 source, 1 pour 2 sources, 3 pour 3 sources, 6 pour 4 sources etc...
# Ordre de remplisage de la correlation_List: de gauche à droite et ligne par ligne, moitié haut-droite de la matrice uniquement, puis symétrie de ces valeurs pour la moitié bas-gauche.
perturbation_parameter_sd = 0.01 # Écart-type de la distribution normale qui génère les erreurs de calibration des capteurs

thetaList = [theta1, theta2]

X = generate_X_matrix(nbSources, nbSensors, nbTimePoints, thetaList, var_ratio, correlation_List, signal_noise_ratio, perturbation_parameter_sd)
estimated_angles = beamforming_method(X, nbSensors, nbSources, print_angles=True, draw_plot=True)