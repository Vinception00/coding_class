import sys
import os

# Ajoute le dossier parent (Tp1) au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_prep import load_data, clean_data
from src.scaling import scale_features
from src.predict import predict
from src.logger import get_logger
from src.features_engineering import process_features
import joblib

logger = get_logger()

if __name__ == "__main__":
    logger.info("Début du script de prédiction.")
    try:
        df = load_data("data/test.csv")
        df = clean_data(df)
        df = process_features(df)
        df = scale_features(df, '.\models\scaler.pkl')
        # Charger les colonnes du modèle entraîné
        train_columns = joblib.load("models/features_columns.pkl")

        # Ajuster les features du jeu de test pour correspondre à celles de l'entraînement
        for col in train_columns:
            if col not in df.columns:
                df[col] = 0  # Ajouter les colonnes manquantes avec des 0

        # Supprimer les colonnes en trop (qui n'existaient pas à l'entraînement)
        df = df[train_columns]
        
        predictions = predict(r".\models\titanic_model.pkl", df)
        logger.success("Prédictions réalisées avec succès.")
        print(predictions)
    except Exception as e:
        logger.error(f"Erreur dans le script de prédiction : {e}")
