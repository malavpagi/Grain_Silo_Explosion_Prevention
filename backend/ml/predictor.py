import keras
import tensorflow as tf
import numpy as np
import requests
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "silo_model_v4_4.keras")

MODEL_URL = os.environ.get("MODEL_URL")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    else:
        raise Exception(
            f"Failed to download model: {response.status_code}"
        )

keras.config.set_floatx('float32')
keras.mixed_precision.set_global_policy('float32')
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_risk(X_test):
    # Ensure correct dtype
    X_test = np.asarray(X_test, dtype=np.float32)

    # Keras inference
    preds = model.predict(X_test, verbose=1)

    risk_score = float(preds[0][0])

    # Risk bucketing
    if risk_score <= 0.0:
        risk_level = "SAFE"
    elif risk_score <= 0.1:
        risk_level = "VERY LOW"
    elif risk_score <= 0.3:
        risk_level = "LOW"
    elif risk_score <= 0.6:
        risk_level = "MEDIUM"
    elif risk_score <= 0.8:
        risk_level = "HIGH"
    else:
        risk_level = "VERY HIGH"

    return risk_score, risk_level
