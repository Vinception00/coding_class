import sys
import os

# Ajoute le dossier parent (Tp1) au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import pandas as pd
from src.logger import get_logger

logger = get_logger()

def scale_features(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Normalise les variables numériques et sauvegarde le scaler.
    """
    try:
        variables = [c  for c in df.columns if c != 'survived']
        scaler = StandardScaler()
        scaler.fit(df[variables])
        df[variables] = scaler.transform(df[variables])
        
        joblib.dump(scaler, output_path)
        logger.info(f"Normalisation terminée et scaler sauvegardé : {output_path}")
        return df
    except Exception as e:
        logger.error(f"Erreur dans la normalisation des features : {e}")
        raise
