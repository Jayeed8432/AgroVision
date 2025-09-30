# train_model.py
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Dummy training data representing features like soil moisture, NDVI, etc.
X = np.array([[0.1, 0.2], [0.2, 0.3], [0.15, 0.25], [0.7, 0.8]])
y = np.array([0, 0, 0, 1])  # 0=healthy, 1=unhealthy

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'ndvi_model.pkl')
print("Model trained and saved as ndvi_model.pkl")
