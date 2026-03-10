import os
import base64
from flask import Flask, request, jsonify
from model_inference import predict_personality

TEMP_DIR = "temp_recordings"
os.makedirs(TEMP_DIR, exist_ok=True)

NUM_QUESTIONS = 5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # allow large uploads


@app.route("/")
def home():
    return "API is running"


@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    videos = data.get("videos", [])

    if len(videos) != NUM_QUESTIONS:
        return jsonify({"error": "Invalid number of videos"}), 400

    paths = []

    # Decode and save videos
    for i, b64 in enumerate(videos):
        try:
            path = os.path.join(TEMP_DIR, f"q{i}.mp4")
            video_bytes = base64.b64decode(b64)

            with open(path, "wb") as f:
                f.write(video_bytes)

            paths.append(path)

        except Exception as e:
            return jsonify({
                "error": f"Failed to decode video {i}",
                "details": str(e)
            }), 400

    # Run model prediction
    try:
        scores = predict_personality(paths)

    except Exception as e:
        return jsonify({
            "error": "Model inference failed",
            "details": str(e)
        }), 500

    # Delete temporary videos after prediction
    for p in paths:
        if os.path.exists(p):
            os.remove(p)

    return jsonify({
        "status": "success",
        "scores": scores
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)