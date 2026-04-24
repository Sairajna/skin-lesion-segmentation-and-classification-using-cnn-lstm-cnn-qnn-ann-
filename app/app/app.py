from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import os
import tempfile
import tensorflow as tf
import joblib

# Load trained model, scaler, and label encoder
model = tf.keras.models.load_model('skin_cancer_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')  # use if you want class names

# Dynamically get timesteps/features_per_timestep from model input
timesteps, features_per_timestep = model.input_shape[1:3]

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_stream):
    with tempfile.NamedTemporaryFile(delete=False) as temp_img:
        temp_img.write(file_stream.read())
        temp_img_path = temp_img.name
    img = Image.open(temp_img_path).convert('RGB')
    os.remove(temp_img_path)
    return img

# Use your actual feature extraction code here (must match training pipeline!)
def extract_features(img):
    # For example: flatten image, or use engineered features from notebook
    arr = np.array(img).flatten()
    # Pad/truncate to correct number of features
    total_features = timesteps * features_per_timestep
    out = np.zeros(total_features)
    out[:min(len(arr), total_features)] = arr[:min(len(arr), total_features)]
    return out

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    try:
        img = preprocess_image(file)
        features = extract_features(img)
        features = scaler.transform([features])
        features = features.reshape(1, timesteps, features_per_timestep)
        y_pred_proba = model.predict(features)
        y_pred = np.argmax(y_pred_proba, axis=1)
        output_class = int(y_pred[0])
        output_name = label_encoder.inverse_transform([output_class])[0]
        output_proba = float(np.max(y_pred_proba))
        return jsonify({
            'predicted_class_index': output_class,
            'predicted_disease': output_name,
            'confidence': output_proba
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
