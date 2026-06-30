# Fonction du mouvement optimal

(u1, u2) = motion_optimal(x, y, theta, K, ubar, R)

où K est le gain, ubar un paramètre, et R le rayon pour le cercle

# Creation d'un dataset d'entrainement

inputs, outputs = get_dataset(N, K, ubar, R)

où inputs est de dimension (N,4) où pour chaque ligne on a: x, y, cos(theta), sin(theta) 

et outputs est de dimension (N,2) où pour chaque ligne on a: u1, u2 

Cette fonction "get_dataset" tire N fois aléatoirement un triplet  (x,y,theta) et utiliser la fonction "motion_optimal" ci dessus pour obtenir les (u,v) optimaux correspondant

La version finale de "get_dataset" que l'on utilise, elle tire N=1 millions de données.

- Dont la moitiée est tirée uniformément dans un carré

- et l'autre moitiée est tirée à partir d'une gaussienne centrée sur le cercle de rayon 10 


# Entrainement d'un modèle Pytorch 

pytorch_model = train_network(inputs, outputs, hyperparameters)

où hyperparameters est la liste de paramètres d'entrainement pour le réseau de Neurone 

# Enregistrement du modèle Pytorch au format csv

register_to_csv(pytorch_model)

