from flask import Flask, request, jsonify, render_template
from mlmodel import CropHealthModel, compute_ndvi, detect_anomaly
import numpy as np

app = Flask(__name__)
model = CropHealthModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    sensor_file = request.files.get('sensor_data')
    image_file = request.files.get('remote_image')
    # Process uploaded files and extract features
    # Placeholder example features (replace with actual parsing)
    features = [0.15, 0.35]

    prediction = model.predict(features)
    ndvi_value = compute_ndvi(0.2, 0.6)
    anomaly = detect_anomaly([ndvi_value])

    response = {
        "prediction": prediction,
        "ndvi": ndvi_value,
        "anomaly_alert": anomaly
    }
    return jsonify(response)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data.get('features')
    prediction = model.predict(features)
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
