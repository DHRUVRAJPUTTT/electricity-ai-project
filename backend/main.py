from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load model and scaler (Removed 'backend/' prefix since Docker runs from inside that folder)
model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.pkl")

# Input schema
class InputData(BaseModel):
    temperature: float
    humidity: float
    hour: int

# 2. Fixed Prediction route
@app.post("/predict")
def predict(data: InputData):
    # Convert input data to a 2D numpy array (which TensorFlow expects)
    input_array = np.array([[data.temperature, data.humidity, data.hour]])

    # Scale the data using the scaler you loaded
    # (If your model was trained on scaled data, you must scale the input or the prediction will be wildly wrong!)
    scaled_input = scaler.transform(input_array)

    # Make the real prediction using the TensorFlow model
    pred = model.predict(scaled_input)

    # Return the AI prediction
    return {"prediction": float(pred)}

# 3. Mount the frontend folder to serve your website
# (This MUST be at the bottom so it doesn't block the /predict route)
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

if __name__ == '__main__':
    # Force the app to run on Hugging Face's mandatory port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)