# ml_model.py
import joblib
import numpy as np

class CropHealthModel:
    def __init__(self, model_path="ndvi_model.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, features):
        features = np.array(features)
        prediction = self.model.predict(features.reshape(1, -1))
        return prediction.tolist()

def compute_ndvi(red_band, nir_band):
    return (nir_band - red_band) / (nir_band + red_band)

def detect_anomaly(ndvi_series):
    return any(ndvi < 0.3 for ndvi in ndvi_series)
