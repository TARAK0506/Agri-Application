import os
import sys
import numpy as np
import tensorflow as tf # type: ignore
from keras.models import load_model as keras_load_model           # type: ignore
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import json
import pickle

# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# print(f"Root directory: {root_dir}")
# sys.path.append(root_dir)
# os.chdir(root_dir)


app = Flask(__name__)

# Get the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
# print(f"Working directory: {working_dir}")

def load_model():
    """Load the models and return them."""
    # Load the Crop Disease Prediction model
    crop_disease_model_path = os.path.join(working_dir, 'Crop_Disease_Prediction_model.h5')
    if not os.path.exists(crop_disease_model_path):
        raise FileNotFoundError(f"Model file not found: {crop_disease_model_path}")
    
    crop_disease_model = keras_load_model(crop_disease_model_path)
    crop_disease_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load the class names for the crop disease prediction model
    class_indices_path = os.path.join(working_dir,  'class_indices.json')
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)

    # Load the Crop Recommendation System model
    crop_recommendation_model_path = os.path.join(working_dir,  'DecisionTree.pkl')
    with open(crop_recommendation_model_path, 'rb') as model_file:
        crop_recommendation_model = pickle.load(model_file)

    return crop_disease_model, class_indices, crop_recommendation_model

# Load models
crop_disease_model, class_indices, crop_recommendation_model = load_model()

# print("Models loaded successfully")

# print("Crop Disease Model: ", crop_disease_model)
# print("Class Indices: ", class_indices)
# print("Crop Recommendation Model: ", crop_recommendation_model)

# Preprocess image for crop disease prediction
def preprocess_image(image):
    image = image.resize((224, 224))  
    image = np.array(image)           
    image = np.expand_dims(image, axis=0) 
    image = image / 255.0             
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open the image and preprocess it
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = crop_disease_model.predict(processed_image)

        # Debugging: print raw predictions
        print(f"Raw predictions: {predictions}")

        # Get the predicted class index and corresponding name
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices[str(predicted_class_index)]

        return jsonify({'prediction': predicted_class_name})

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error for debugging
        return jsonify({'error': str(e)}), 500

# Route for Crop Recommendation
@app.route('/recommend', methods=['POST'])
def recommend_crop():
    try:
        # Get the input data directly from the request
        data = request.get_json()
        print(f"Received data: {data}")

        # Validate input data
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        # Extract features for prediction
        feature_values = [
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]

        # Predict the recommended crop using the loaded model
        recommended_crop = crop_recommendation_model.predict([feature_values])[0]
        return jsonify({'recommended_crop': recommended_crop})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    
    



