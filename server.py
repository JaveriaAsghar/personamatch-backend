# server.py

import os
import base64
from flask import Flask, request, jsonify

TEMP_DIR = "temp_recordings"
os.makedirs(TEMP_DIR, exist_ok=True)

NUM_QUESTIONS = 5

app = Flask(__name__)


@app.route("/")
def home():
    return "API is running"

@app.route("/analyze", methods=["POST"])
def analyze():

    from model_inference import predict_personality  # move import here

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400
    


    videos = data.get("videos", [])

    if len(videos) != NUM_QUESTIONS:
        return jsonify({"error": "Invalid number of videos"}), 400

    paths = []

    for i, b64 in enumerate(videos):
        path = os.path.join(TEMP_DIR, f"q{i}.mp4")
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        paths.append(path)

    scores = predict_personality(paths)

    return jsonify({
        "status": "success",
        "scores": scores
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
