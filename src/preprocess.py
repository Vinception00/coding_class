import sys
import os

# Ajoute le dossier parent (Tp1) au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_prep import load_data, clean_data
from src.features_engineering import process_features
from src.scaling import scale_features
from src.logger import get_logger

logger = get_logger()

if __name__ == "__main__":
    logger.info("Début du prétraitement des données.")
    
    try:
        # Chargement des données
        df = load_data("data/train.csv")
        
        # Nettoyage des données
        df = clean_data(df)

        # Feature engineering
        df = process_features(df)

        # Normalisation
        df = scale_features(df, ".\scripts\scaler.pkl")

        logger.success("Prétraitement des données terminé avec succès.")
    except Exception as e:
        logger.error(f"Erreur pendant le prétraitement : {e}")
