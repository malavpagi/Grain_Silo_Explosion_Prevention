from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import json
from simulation.generator import generate_test_sequence
from ml.predictor import predict_risk

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=["GET"])
@app.route("/home", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory('static', path)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        
        if not payload:
            return jsonify({"error": "No data provided"}), 400
        
        mode = payload.get("mode")

        if mode == "synthetic":
            print("🔄 Generating synthetic test sequence...")
            X_test, viz = generate_test_sequence(return_viz=True)
            print(f"✅ Generated sequence shape: {X_test.shape}")

        elif mode == "user":
            print("📁 Processing user-uploaded sequence...")
            sequence_data = payload.get("sequence")
            
            if not sequence_data:
                return jsonify({"error": "No sequence data provided"}), 400
            
            try:
                # User sends raw temperature data (50 timesteps of 20x20x20)
                raw_sequence = np.array(sequence_data, dtype=np.float32)
                print(f"📊 User sequence shape: {raw_sequence.shape}")
                
                # Expected: (50, 20, 20, 20) or (1, 50, 20, 20, 20, 1)
                if raw_sequence.ndim == 4:
                    # Shape: (50, 20, 20, 20)
                    history = raw_sequence
                elif raw_sequence.ndim == 6:
                    # Shape: (1, 50, 20, 20, 20, 1)
                    history = raw_sequence[0, :, :, :, :, 0]
                else:
                    return jsonify({"error": f"Invalid sequence dimensions: {raw_sequence.shape}. Expected (50, 20, 20, 20) or (1, 50, 20, 20, 20, 1)"}), 400
                
                # Generate visualization metrics
                viz = generate_viz_from_sequence(history)
                
                # Normalize for model (same as generator.py)
                from simulation import Config
                norm_history = (history - Config.AMBIENT) / (350.0 - Config.AMBIENT)
                X_test = norm_history[np.newaxis, ..., np.newaxis]
                print(f"✅ Normalized sequence shape: {X_test.shape}")
                
            except Exception as e:
                print(f"❌ Error processing user sequence: {str(e)}")
                import traceback
                traceback.print_exc()
                return jsonify({"error": f"Invalid sequence format: {str(e)}"}), 400

        else:
            return jsonify({"error": "Invalid mode. Use 'synthetic' or 'user'"}), 400

        # Model inference
        print("🤖 Running model inference...")
        risk_score, risk_level = predict_risk(X_test)
        print(f"📈 Prediction: {risk_level} ({risk_score:.4f})")

        response = {
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "frames": viz["frames"].tolist() if isinstance(viz["frames"], np.ndarray) else viz["frames"],
            "max_temp_log": viz["max_temp_log"],
            "cluster_count_log": viz["cluster_count_log"]
        }

        print(f"✅ Response prepared. Frames: {len(response['frames'])}, Max temp: {max(response['max_temp_log']):.2f}°C")
        return jsonify(response)

    except Exception as e:
        print(f"❌ Error in /predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


def generate_viz_from_sequence(history):
    """
    Generate visualization data from user-uploaded sequence.
    history shape: (50, 20, 20, 20)
    """
    try:
        timesteps = history.shape[0]
        
        max_temp_log = []
        cluster_count_log = []
        frames = []
        
        for t in range(timesteps):
            frame = history[t]
            frames.append(frame.tolist())
            
            # Max temperature
            max_temp = float(np.max(frame))
            max_temp_log.append(max_temp)
            
            # Simple cluster detection (temp > 100°C threshold)
            hot_spots = frame > 100
            cluster_count = int(np.sum(hot_spots) / 8) if np.sum(hot_spots) > 0 else 0
            cluster_count_log.append(cluster_count)
        
        return {
            "frames": frames,
            "max_temp_log": max_temp_log,
            "cluster_count_log": cluster_count_log
        }
    
    except Exception as e:
        print(f"❌ Error generating viz: {str(e)}")
        return {
            "frames": [],
            "max_temp_log": [],
            "cluster_count_log": []
        }


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "SiloGuard AI Backend Running",
        "model_loaded": True
    })


if __name__ == "__main__":
    print("🚀 Starting SiloGuard AI Backend...")
    print("📍 Server running on http://localhost:5000")
    print("📊 Model loaded from:", "model/silo_model_v4_4.keras")
    app.run(debug=True, host='0.0.0.0', port=5000)