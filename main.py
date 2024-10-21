import io
import os
import json
import pickle

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image

import tensorflow as tf # type: ignore
from keras.models import load_model # type: ignore


from fastapi import FastAPI, UploadFile, File, HTTPException, Request # type: ignore
from fastapi.responses import HTMLResponse, JSONResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi.templating import Jinja2Templates # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore


# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI()

# Add CORS Middleware
# origins = [
#     "http://localhost:3000",  
#     "http://127.0.0.1:3000",
#     # Add other origins as needed
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the Crop Disease Prediction model
working_dir = os.path.dirname(os.path.abspath(__file__))
crop_disease_model_path = os.path.join(working_dir, 'Models', 'Crop Disease Prediction model.h5')
crop_disease_model = load_model(crop_disease_model_path)

# Load the class names for the crop disease prediction model
class_indices_path = os.path.join(working_dir, 'class_indices.json')
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Load the Crop Recommendation System model
crop_recommendation_model_path = os.path.join(working_dir, 'Models', 'DecisionTree.pkl')
with open(crop_recommendation_model_path, 'rb') as model_file:
    crop_recommendation_model = pickle.load(model_file)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def preprocess_image(image):
    
    image = image.resize((224, 224))  
    image_array = np.array(image)    
    image_array = image_array / 255.0 
    image_array = np.expand_dims(image_array, axis=0)  
    return image_array

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Crop Disease Prediction Route
@app.post("/predict")
async def predict_disease(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    try:
        # Read and process the image
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image) 
        
        # Predict the crop disease
        predictions = crop_disease_model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices[str(predicted_class_index)]

        # Return the prediction result as JSON
        return JSONResponse(content={"prediction": predicted_class_name})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")


# Route for Crop Recommendation
@app.post("/recommend")
async def recommend_crop(data: dict):
    try:
        
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing field: {field}")

        
        features = [
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]
        
        
        recommended_crop = crop_recommendation_model.predict([features])[0]
        
        return JSONResponse(content={'recommended_crop': recommended_crop})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8000)
