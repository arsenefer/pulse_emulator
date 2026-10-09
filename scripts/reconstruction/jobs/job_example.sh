#!/bin/bash

# SLURM options:

#SBATCH --job-name=final_recons_optim    # Nom du job
#SBATCH --output=../logs/final_recons_optim_%j.log   # Standard output et error log

#SBATCH --partition=htc               # Choix de partition (htc par défaut)

#SBATCH --ntasks=1                    # Exécuter une seule tâche
#SBATCH --mem=120000                   # Mémoire en MB par défaut
#SBATCH --time=0-24:00:00             # Délai max = 7 jours
#SBATCH --cpus-per-task=120
#SBATCH --mail-type=END,FAIL          # Ivénements déclencheurs (NONE, BEGIN, END, FAIL, ALL)

#SBATCH --licenses=sps                # Déclaration des ressources de stockage et/ou logicielles
# Commandes à soumettre :

source [ENV]
echo "$@"
python scripts/reconstruction/OPTIMIZER_final.py "$@"    