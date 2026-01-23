import keras
import tensorflow as tf
import numpy as np

MODEL_PATH = "C:/Users/malav/My_Storage/Projects/Grain_Silo_Explosion_Prevention/backend/model/silo_model_v4_4.keras"
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
