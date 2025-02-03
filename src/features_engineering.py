import sys
import os

# Ajoute le dossier parent (Tp1) au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import numpy as np
import pandas as pd
from src.logger import get_logger

logger = get_logger()

def get_first_cabin(row: str) -> str:
    """Extrait la première cabine d’un passager. Renvoie 'Missing' si la cabine est inconnue."""
    try:
        if pd.isna(row) or not isinstance(row, str):  # Vérifie si NaN ou non string
            return "Missing"
        return row.split()[0]  # Prend la première cabine
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction de la cabine : {e} (Valeur reçue: {row})")
        return "Missing"

def get_title(passenger: str) -> str:
    """Extrait le titre d’un passager."""
    try:
        if re.search('Mrs', passenger):
            return 'Mrs'
        elif re.search('Mr', passenger):
            return 'Mr'
        elif re.search('Miss', passenger):
            return 'Miss'
        elif re.search('Master', passenger):
            return 'Master'
        else:
            return 'Other'
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du titre : {e}")
        return 'Other'

def find_frequent_labels(df: pd.DataFrame, var: str, rare_perc: float):
    """Trouve les labels fréquents dans une variable catégorielle."""
    try:
        freq_labels = df[var].value_counts(normalize=True)
        return freq_labels[freq_labels > rare_perc].index.tolist()
    except Exception as e:
        logger.error(f"Erreur dans la détection des labels fréquents : {e}")
        raise

def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applique toutes les transformations de features."""
    try:
        # Extraction des features "cabin" et "title"
        df['cabin'] = df['cabin'].apply(get_first_cabin)
        df['title'] = df['name'].apply(get_title)
        df['cabin'] = df['cabin'].str[0]  # Capture la première lettre
        
        vars_cat = df.select_dtypes(include=['object']).columns.tolist()
        
        # Gestion des valeurs manquantes pour variables catégorielles
        df[vars_cat] = df[vars_cat].fillna('Missing')

        # Remplacement des valeurs rares par "Rare"
        for var in vars_cat:
            frequent_labels = find_frequent_labels(df, var, 0.05)
            df[var] = np.where(df[var].isin(frequent_labels), df[var], 'Rare')

        # Ajout d’indicateurs de valeurs manquantes pour variables numériques
        for var in ['age', 'fare']:
            df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
            df[var].fillna(df[var].median(), inplace=True)

        #Encodage des variables catégorielles 
        df = pd.get_dummies(df, columns=vars_cat, drop_first=True)

        logger.info("Feature engineering terminé.")
        return df
    except Exception as e:
        logger.error(f"Erreur dans le traitement des features : {e}")
        raise