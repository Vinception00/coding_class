import joblib
import pandas as pd

def predict(model_path: str, df: pd.DataFrame) -> pd.Series:
    """
    Charge le modèle et prédit les survivants.
    """
    model = joblib.load(model_path)
    return model.predict(df)
