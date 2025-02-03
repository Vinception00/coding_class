import sys
import os

# Ajoute le dossier parent (Tp1) au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger()

def load_data(file_path: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Chargement réussi : {file_path} - {df.shape[0]} lignes, {df.shape[1]} colonnes.")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données : {e}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données en remplaçant les valeurs '?' par NaN et en mettant en minuscules les colonnes.
    """
    try:
        df = df.replace('?', np.nan)
        df.columns = df.columns.str.lower()
        df['fare'] = df['fare'].astype('float')
        df['age'] = df['age'].astype('float')
        df.drop(labels=['ticket'], axis=1, inplace=True)
        logger.info("Nettoyage des données terminé.")
        return df
    except Exception as e:
        logger.error(f"Erreur dans le nettoyage des données : {e}")
        raise
