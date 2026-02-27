# feature_extraction.py

import os
import cv2
import math
import numpy as np
import librosa
import whisper
import torch
import mediapipe as mp
from moviepy import VideoFileClip
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
AUDIO_DIM = 68
VISUAL_DIM = 48
TEXT_DIM = 384

# =========================
# LOAD MODELS (ONCE)
# =========================
whisper_model = whisper.load_model("tiny")

sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
if torch.cuda.is_available():
    sentence_model = sentence_model.cuda()
sentence_model.eval()

mp_face_mesh = mp.solutions.face_mesh


# =========================
# HELPER FUNCTIONS
# =========================
def estimate_head_pose(landmarks):
    focal_length = 640
    center = (320, 240)

    cam_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1))

    model_points = np.array([
        (0, 0, 0),
        (0, -330, -65),
        (-225, 170, -135),
        (225, 170, -135),
        (-150, -150, -125),
        (150, -150, -125)
    ]) / 4.5

    image_points = np.array([
        (landmarks[1].x, landmarks[1].y),
        (landmarks[175].x, landmarks[175].y),
        (landmarks[33].x, landmarks[33].y),
        (landmarks[263].x, landmarks[263].y),
        (landmarks[61].x, landmarks[61].y),
        (landmarks[291].x, landmarks[291].y)
    ], dtype=np.float32) * [640, 480]

    success, rvec, _ = cv2.solvePnP(
        model_points, image_points, cam_matrix, dist_coeffs
    )

    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)

    pitch = math.degrees(math.asin(rmat[2, 1]))
    yaw = math.degrees(math.atan2(-rmat[2, 0], rmat[2, 2]))
    roll = math.degrees(math.atan2(-rmat[0, 1], rmat[1, 1]))

    return pitch, yaw, roll


# =========================
# FEATURE EXTRACTION
# =========================
def extract_visual_features(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros(VISUAL_DIM, dtype=np.float32)

    head_poses, smiles, mouth_open, motion = [], [], [], []
    prev_center = None

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_mesh:

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 3 != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                pitch, yaw, roll = estimate_head_pose(lm)
                head_poses.append([pitch, yaw, roll])

                l_c = np.array([lm[61].x, lm[61].y])
                r_c = np.array([lm[291].x, lm[291].y])
                l_e = np.array([lm[33].x, lm[33].y])
                r_e = np.array([lm[263].x, lm[263].y])

                iod = np.linalg.norm(l_e - r_e) + 1e-6
                smiles.append(np.linalg.norm(l_c - r_c) / iod)

                mouth_open.append(lm[14].y - lm[13].y)

                center = np.mean([[p.x, p.y] for p in lm], axis=0)
                if prev_center is not None:
                    motion.append(np.linalg.norm(center - prev_center))
                prev_center = center

            frame_idx += 1

    cap.release()

    def stats(x):
        if len(x) == 0:
            return np.zeros(8)
        x = np.array(x)
        return np.array([
            np.mean(x), np.std(x), np.max(x), np.min(x),
            np.percentile(x, 90), np.percentile(x, 10),
            np.ptp(x), min(len(x) / 100.0, 20.0)
        ])

    feats = np.concatenate([
        stats([p[0] for p in head_poses]),
        stats([p[1] for p in head_poses]),
        stats([p[2] for p in head_poses]),
        stats(smiles),
        stats(mouth_open),
        stats(motion)
    ])

    return feats.astype(np.float32)


def extract_audio_features(video_path):
    try:
        clip = VideoFileClip(video_path)
        audio_path = "temp.wav"
        clip.audio.write_audiofile(audio_path, logger=None)

        y, sr = librosa.load(audio_path, sr=None)
        os.remove(audio_path)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        feats = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(zcr, axis=1), np.std(zcr, axis=1),
            np.mean(rms, axis=1), np.std(rms, axis=1)
        ])

        return feats.astype(np.float32)

    except Exception:
        return np.zeros(AUDIO_DIM, dtype=np.float32)


def extract_text_features(video_path):
    try:
        result = whisper_model.transcribe(video_path, fp16=False)
        text = result["text"].strip()

        if len(text) < 5:
            return np.zeros(TEXT_DIM, dtype=np.float32)

        with torch.no_grad():
            emb = sentence_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

        return emb.astype(np.float32)

    except Exception:
        return np.zeros(TEXT_DIM, dtype=np.float32)
