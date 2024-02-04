import numpy as np
import matplotlib.pyplot as plt
import cmath
import math

#fonction qui gènere un signal aléatoire sous forme d'un nombre complexe, la distribution du signal suit une loi gaussienne
def generer_signal_complexe():
    # Définir les paramètres de la distribution gaussienne
    mu = 0  # Moyenne
    sigma = 10  # Écart-type

    #Générer des nombres aléatoires selon une distribution gaussienne
    x = np.random.normal(mu, sigma)
    y = np.random.normal(mu, sigma)
    z = complex(x,y)
    return z


#fonction qui gènere le signal reçus en ajoutant le retard exponentiel et le biais

def generer_signal():
    ncapteur=10
    nsignal=1
    teta=[]
    for e in range(nsignal):
        teta.append(0)
    d=10
    c=299792458
    tau=[]
    for e in range(nsignal):
        tau.append(d*cmath.sin(teta[e])/c)
    steering=np.empty((ncapteur, nsignal),dtype=complex)
    #on crée les coefficients de la matrice des steering vectors
    for capteur in range(ncapteur):
        for signal in range(nsignal):
            steering[capteur,signal]=cmath.exp(-2j*cmath.pi*tau[signal]*capteur)
    
    #on génère les signaux
    signaux=np.empty((nsignal,1),dtype=complex)
    for signal in range(nsignal):
        signaux[signal,0]=generer_signal_complexe()
    
    #On crée la matrice de biais
    biais=np.empty((ncapteur,1),dtype=complex)
    for capteur in range(ncapteur):
        biais[capteur,0]=generer_signal_complexe()
        
    x=np.dot(steering,signaux)+10**-3*biais
    return x

x=generer_signal()   
print(x)       
        
def beamforming(x):
    d=10
    c=299792458
    
    angles = np.linspace(0, np.pi, 180)  # Génère des angles de 0 à pi radians
    tau=[]
    for e in range(len(angles)):
        tau.append(d*cmath.sin(angles[e])/c)
        
        
    max=0
    angle_max=0
    poids_optimal=np.empty((len(x),1),dtype=complex)
    
    for angle in range(len(angles)):
        poids=np.empty((len(x),1),dtype=complex)
        for capteur in range(len(x)):
            poids[capteur,0]=cmath.exp(2j*cmath.pi*tau[angle]*capteur)
        
        puissance=np.abs(np.sum(poids * x))
        if puissance>=max:
            max=puissance
            angle_max=angles[angle]
            poids_optimal=poids
        
    # Afficher le diagramme de formation de voies
    
    #liste des signaux possibles
    liste_puissance=[]
    for e in range(len(angles)):
        steering=np.empty((len(x),1),dtype=complex)
        tau=d*cmath.sin(angles[e])/c
        for capteur in range(len(x)):
            steering[capteur,0]=cmath.exp(-2j*cmath.pi*tau*capteur)
        #print(poids_optimal*steering)
        p=np.abs(np.sum(poids_optimal * steering))
        liste_puissance.append(p)
    intensite_normalisee = liste_puissance/np.linalg.norm(liste_puissance)
    #print(steering)
    print(intensite_normalisee)
    plt.figure()
    plt.plot(np.degrees(angles), intensite_normalisee)
    plt.title("Diagramme de Formation de Voies")
    plt.xlabel("Angle (degrés)")
    plt.ylabel("Intensité")
    plt.show() 
    
    
x=generer_signal()
beamforming(x)
