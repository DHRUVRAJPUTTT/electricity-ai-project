from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import uvicorn
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("=== MODEL INFO ===")
print("Input shape:", model.input_shape)   # expect (None, 24, 20)
print("Output shape:", model.output_shape)
print("==================")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    temperature: float
    humidity: float
    hour: int

@app.post("/predict")
def predict(data: InputData):
    try:
        now = datetime.now()
        hour        = data.hour
        month       = now.month
        day_of_week = now.weekday()

        # Season flags
        if month in [9, 10, 11]:   season = "Fall"
        elif month in [3, 4, 5]:   season = "Spring"
        elif month in [6, 7, 8]:   season = "Summer"
        else:                       season = "Winter"

        season_fall   = 1 if season == "Fall"   else 0
        season_spring = 1 if season == "Spring" else 0
        season_summer = 1 if season == "Summer" else 0
        season_winter = 1 if season == "Winter" else 0

        # Cyclical hour encoding
        hour_sin = float(np.sin(2 * np.pi * hour / 24))
        hour_cos = float(np.cos(2 * np.pi * hour / 24))

        # Approximate dew point
        dew_point = data.temperature - ((100 - data.humidity) / 5.0)

        # Midpoint defaults for lag/unknown features
        lag_mid    = (19217.76 + 76633.03) / 2.0
        wind_speed = 15.0
        pressure   = 1007.45

        # Build one row of 20 features (same order scaler was trained on)
        one_row = np.array([[
            lag_mid,           # [0]  Load Demand (placeholder)
            lag_mid,           # [1]  lag_1
            lag_mid,           # [2]  lag_2
            lag_mid,           # [3]  lag_3
            lag_mid,           # [4]  lag_4
            lag_mid,           # [5]  lag_24
            lag_mid,           # [6]  lag_168
            data.temperature,  # [7]  Temperature (°C)
            dew_point,         # [8]  Dew Point (°C)
            data.humidity,     # [9]  Humidity (%)
            wind_speed,        # [10] Wind Speed (km/h)
            pressure,          # [11] Pressure (hPa)
            hour_sin,          # [12] hour_sin
            hour_cos,          # [13] hour_cos
            day_of_week,       # [14] Day of Week
            month,             # [15] Month
            season_fall,       # [16] Season_Fall
            season_spring,     # [17] Season_Spring
            season_summer,     # [18] Season_Summer
            season_winter,     # [19] Season_Winter
        ]], dtype=np.float64)  # shape: (1, 20)

        # Scale the single row
        scaled_row = scaler.transform(one_row)  # shape: (1, 20)

        # --- KEY FIX ---
        # LSTM expects (batch, timesteps, features) = (1, 24, 20)
        # Tile the scaled row across 24 timesteps
        lstm_input = np.tile(scaled_row, (24, 1))          # shape: (24, 20)
        lstm_input = lstm_input.reshape(1, 24, 20)          # shape: (1, 24, 20)

        pred = model.predict(lstm_input, verbose=0)
        print(f"pred raw: {pred}, shape: {pred.shape}")

        prediction_scaled = float(pred[0][0])

        # Inverse-scale back to kW (Load Demand = column 0)
        prediction_kw = (prediction_scaled - float(scaler.min_[0])) / float(scaler.scale_[0])
        print(f"Prediction kW: {prediction_kw}")

        return {"prediction": round(float(prediction_kw), 2)}

    except Exception as e:
        import traceback
        print(f"PREDICTION ERROR: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}

app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=7860)
