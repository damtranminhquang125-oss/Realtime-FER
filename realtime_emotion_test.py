"""
Test realtime nhận diện cảm xúc khuôn mặt qua webcam.
"""

import time
from collections import deque

import cv2
import keras
import numpy as np
import tensorflow as tf
from model_architecture import custom_cnn, SpatialAttention
from tensorflow.keras.models import load_model


# =========================
# CONFIG
# =========================
MODEL_PATH = "best_model.keras"

CLASS_LABELS = [
    "Angry", "Disgust", "Fear", "Happy",
    "Sad", "Surprise", "Neutral"
]

IMG_SIZE = 48
CAMERA_INDEX = 0
MARGIN_RATIO = 0.18
DETECT_EVERY_N_FRAMES = 5
SMOOTHING_WINDOW = 7

# =========================
# LOAD EMOTION MODEL
# =========================
print("Dang tai model...")
model = load_model(
    "best_model.keras",
    custom_objects={"SpatialAttention": SpatialAttention},
    compile=False
)
print("Emotion model loaded.")

import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,  # 0: khuôn mặt gần, 1: xa hơn
    min_detection_confidence=0.6
)


def get_largest_face_bbox(frame):
    frame_h, frame_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_detector.process(rgb_frame)

    if not result.detections:
        return None

    best_detection = max(
        result.detections,
        key=lambda detection: detection.location_data.relative_bounding_box.width
        * detection.location_data.relative_bounding_box.height
    )

    box = best_detection.location_data.relative_bounding_box

    x = int(box.xmin * frame_w)
    y = int(box.ymin * frame_h)
    w = int(box.width * frame_w)
    h = int(box.height * frame_h)

    x = max(0, x)
    y = max(0, y)
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)

    return x, y, w, h


def square_face_crop(frame, bbox, margin_ratio=MARGIN_RATIO):
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = bbox

    center_x = x + w / 2
    center_y = y + h / 2
    side = int(max(w, h) * (1 + 2 * margin_ratio))

    x1 = max(0, int(center_x - side / 2))
    y1 = max(0, int(center_y - side / 2))
    x2 = min(frame_w, int(center_x + side / 2))
    y2 = min(frame_h, int(center_y + side / 2))

    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def preprocess_face(face_bgr):
    # Khớp hoàn toàn pipeline train: gray -> 48x48 -> /255.
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    tensor = gray.astype(np.float32) / 255.0
    return tensor[np.newaxis, ..., np.newaxis]


# =========================
# REALTIME
# =========================
print("Đang mở camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Không mở được camera.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prob_history = deque(maxlen=SMOOTHING_WINDOW)

frame_count = 0
last_bbox = None
last_detected_frame = 0
previous_time = time.time()

print("Nhấn Q để thoát.")

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    # Không detect tất cả frame để tăng FPS.
    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        detected_bbox = get_largest_face_bbox(frame)

        if detected_bbox is not None:
            last_bbox = detected_bbox
            last_detected_frame = frame_count

    # Xóa bbox cũ nếu đã mất mặt lâu.
    if frame_count - last_detected_frame > DETECT_EVERY_N_FRAMES * 5:
        last_bbox = None
        prob_history.clear()

    if last_bbox is not None:
        face_crop, draw_bbox = square_face_crop(frame, last_bbox)

        if face_crop is not None and face_crop.size > 0:
            face_tensor = preprocess_face(face_crop)

            # Không dùng TTA.
            probability = model(face_tensor, training=False).numpy()[0]

            prob_history.append(probability)
            smooth_probability = np.mean(prob_history, axis=0)
            
            canvas = np.zeros((280, 380, 3), dtype=np.uint8)

            for i, (emotion, prob) in enumerate(zip(CLASS_LABELS, smooth_probability)):
                y = i * 38 + 15
                bar_width = int(prob * 260)

                cv2.rectangle(canvas, (110, y), (110 + bar_width, y + 24), (0, 180, 255), -1)
                cv2.putText(
                    canvas,
                    emotion,
                    (10, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                cv2.putText(
                    canvas,
                    f"{prob * 100:.1f}%",
                    (285, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            cv2.imshow("Probabilities", canvas)

            label_id = int(np.argmax(smooth_probability))
            confidence = float(smooth_probability[label_id])

            x1, y1, x2, y2 = draw_bbox
            color = (0, 220, 0) if confidence >= 0.45 else (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{CLASS_LABELS[label_id]}: {confidence:.0%}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )

    current_time = time.time()
    fps = 1 / max(current_time - previous_time, 1e-6)
    previous_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Realtime Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()