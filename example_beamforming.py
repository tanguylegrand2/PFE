from generate_signal import generate_X_matrix
from beamforming import beamforming_method

nbSources = 2 # Nombre de sources
nbSensors = 10 # Nombre de capteurs
nbTimePoints = 100 # Nombre de points temporels
signal_noise_ratio = 3 # Rapport signal sur bruit en décibels. Si 'False', cela revient à une absence totale de bruit.
theta1 = 5 # Angle entre la perpendiculaire à la ligne de capteurs, et la source 1
theta2 = -5 # Angle entre la perpendiculaire à la ligne de capteurs, et la source 2
var1 = 1 # Variance du signal 1
var2 = 1 # Variance du signal 2
correlation_List = [0.4] # Liste des corrélations. Il y a une corrélation nécéssaire pour chaque paire distincte de sources différentes: 0 pour 1 source, 1 pour 2 sources, 3 pour 3 sources, 6 pour 4 sources etc...
# Ordre de remplisage de la correlation_List: de gauche à droite et ligne par ligne, moitié haut-droite de la matrice uniquement, puis symétrie de ces valeurs pour la moitié bas-gauche.

thetaList = [theta1, theta2]
varList = [var1, var2]

X = generate_X_matrix(nbSources, nbSensors, nbTimePoints, thetaList, varList, correlation_List, signal_noise_ratio)
estimated_angles = beamforming_method(X, nbSensors, nbSources, print_angles=True, draw_plot=True)