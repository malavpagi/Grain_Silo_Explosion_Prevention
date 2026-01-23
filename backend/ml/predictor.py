import numpy as np
import tensorflow as tf

def predict_risk(X_test):
    # Ensure correct dtype
    MODEL_PATH = "C:/Users/malav/My_Storage/Projects/Grain_Silo_Explosion_Prevention/backend/model/silo_model_v4_2.keras"
    X_test = np.asarray(X_test, dtype=np.float32)

    model = tf.keras.models.load_model(MODEL_PATH)
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
    elif risk_score <= 0.5:
        risk_level = "MEDIUM"
    elif risk_score <= 0.8:
        risk_level = "HIGH"
    else:
        risk_level = "VERY HIGH"

    return risk_score, risk_level
