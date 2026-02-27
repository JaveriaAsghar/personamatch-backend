# model_inference.py

import numpy as np
from tensorflow.keras.models import load_model
from feature_extraction import (
    extract_audio_features,
    extract_visual_features,
    extract_text_features,
    AUDIO_DIM, VISUAL_DIM, TEXT_DIM
)

MODEL_PATH = "fyp_model.keras"
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

model = load_model(MODEL_PATH)


def predict_personality(video_paths):
    audio_feats, visual_feats, text_feats = [], [], []

    for path in video_paths:
        audio_feats.append(extract_audio_features(path))
        visual_feats.append(extract_visual_features(path))
        text_feats.append(extract_text_features(path))

    audio_feats = np.array(audio_feats)
    visual_feats = np.array(visual_feats)
    text_feats = np.array(text_feats)

    # Mean aggregation
    final_audio = audio_feats.mean(axis=0)
    final_visual = visual_feats.mean(axis=0)
    final_text = text_feats.mean(axis=0)

    pred = model.predict([
        final_audio.reshape(1, AUDIO_DIM),
        final_visual.reshape(1, VISUAL_DIM),
        final_text.reshape(1, TEXT_DIM)
    ], verbose=0)[0]

    # Convert 0–1 → 1–10
    scores = (pred * 9) + 1

    return {trait: float(score) for trait, score in zip(TRAITS, scores)}
