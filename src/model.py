import sys
import os

# Ajoute le dossier parent (Tp1) au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from src.logger import get_logger

logger = get_logger()

def train_model(df: pd.DataFrame, target: str, model_path: str):
    """
    Entraîne un modèle de classification et le sauvegarde.
    """
    try:
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        model = LogisticRegression(C=0.0005, random_state=0)        
        model.fit(X_train, y_train)

        joblib.dump(model, model_path)
        joblib.dump(list(X_train.columns), "models/features_columns.pkl")
        logger.info(f"Modèle entraîné et sauvegardé à {model_path}. Score : {model.score(X_test, y_test):.4f}")
        return model
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle : {e}")
        raise
