from flask import Flask, jsonify, request
from flask_cors import CORS

from simulation.generator import generate_silo_sequence
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

    if mode == "user":
        sequence = payload.get("sequence")

        if sequence is None:
            return jsonify({
                "error": "User data selected but no sequence provided"
            }), 400

    elif mode == "synthetic":
        sequence = generate_silo_sequence().tolist()

    else:
        return jsonify({
            "error": "Invalid mode. Use 'user' or 'synthetic'"
        }), 400

    risk_score, risk_level = predict_risk(sequence)

    return jsonify({
        "mode_used": mode,
        "risk_score": round(risk_score, 3),
        "risk_level": risk_level
    })


if __name__ == "__main__":
    app.run(debug=True)
