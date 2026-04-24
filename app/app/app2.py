import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# --- Initialize Flask application ---
app = Flask(__name__)

# --- Load all necessary files ---
try:
    model = load_model('skin_cancer_model.h5')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    # **NEW**: Load the feature dataset to look up image features
    features_df = pd.read_csv('lesion_features.csv')
    # Set 'image_id' as the index for easy lookup
    features_df.set_index('image_id', inplace=True)
    
    print("✅ Model, preprocessors, and feature data loaded successfully!")

except Exception as e:
    print(f"❌ Error loading files: {e}")
    # In case of an error, set all to None so the app can report it
    model = scaler = label_encoder = features_df = None

# --- Define routes ---
@app.route('/')
def index():
    # Render the main HTML page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if any(x is None for x in [model, scaler, label_encoder, features_df]):
        return jsonify({'error': 'Backend model/data not loaded properly. Check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if file:
        try:
            # Get the image ID from the filename (e.g., "ISIC_0029099.jpg" -> "ISIC_0029099")
            image_id = os.path.splitext(file.filename)[0]

            # **THE FIX IS HERE:** Look up the pre-extracted features from the CSV
            if image_id not in features_df.index:
                return jsonify({'error': f'Features for image {image_id} not found in the dataset.'}), 404

            # Get the feature vector (it's a row in the DataFrame)
            feature_vector = features_df.loc[image_id].drop('diagnosis').values.reshape(1, -1)
            
            # 1. Scale the features
            scaled_features = scaler.transform(feature_vector)
            
            # 2. Reshape for the model input (e.g., to (1, 5, 13))
            timesteps = 5
            features_per_timestep = 13
            model_input = scaled_features.reshape(1, timesteps, features_per_timestep)

            # 3. Make prediction
            prediction_probs = model.predict(model_input)
            
            # 4. Decode the prediction
            predicted_class_index = np.argmax(prediction_probs, axis=1)[0]
            confidence = np.max(prediction_probs) * 100
            predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]

            # Return the result
            return jsonify({
                'prediction': predicted_class_name,
                'confidence': f'{confidence:.2f}'
            })

        except Exception as e:
            # This will catch errors like the image_id not being in the CSV
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    return jsonify({'error': 'An unknown error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)