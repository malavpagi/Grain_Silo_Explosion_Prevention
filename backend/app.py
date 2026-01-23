from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from simulation.generator import generate_test_sequence
from ml.predictor import predict_risk

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
@app.route("/home", methods=["GET"])
def home():
    return jsonify({
        "message": "Grain Silo Explosion Risk API running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    payload = request.get_json()
    mode = payload.get("mode")

    if mode == "synthetic":
        X_test, viz = generate_test_sequence(return_viz=True)

    elif mode == "user":
        # user sends already-normalized sequence
        X_test = np.array(payload["sequence"], dtype=np.float32)
        viz = None

    else:
        return jsonify({"error": "Invalid mode"}), 400

    # Model inference
    risk_score, risk_level = predict_risk(X_test)

    response = {
        "risk_score": round(risk_score, 3),
        "risk_level": risk_level
    }

    if viz:
        response.update({
            "frames": viz["frames"].tolist(),
            "max_temp_log": viz["max_temp_log"],
            "cluster_count_log": viz["cluster_count_log"]
        })

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
