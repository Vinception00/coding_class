import sys
import os

# Ajoute le dossier parent (Tp1) au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_prep import load_data, clean_data
from src.scaling import scale_features
from src.model import train_model
from src.logger import get_logger
from src.features_engineering import process_features

logger = get_logger()

if __name__ == "__main__":
    logger.info("Début du script d'entraînement.")
    try:
        df = load_data("data/train.csv")
        df = clean_data(df)
        df = process_features(df)
        df = scale_features(df,'.\models\scaler.pkl')
        train_model(df, "survived", r".\models\titanic_model.pkl")
        logger.success("Entraînement terminé avec succès.")
    except Exception as e:
        logger.error(f"Erreur dans le script d'entraînement : {e}")
