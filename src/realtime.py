"""
realtime.py

Real-time demo that:
- captures webcam frames
- uses MediaPipe Hands to detect/track a hand (for robust bounding box)
- crops the hand region, preprocesses, runs classification model
- debounces predictions and triggers mapped actions

Usage:
  python realtime.py --model models/final_model.h5 --class_map models/class_indices.json
"""

import cv2
import time
import argparse
import numpy as np
import mediapipe as mp
import tensorflow as tf

from src.utils import preprocess_image_for_model, load_class_names, center_pad_bbox
from src.action_mapper import ActionMapper

def main(model_path, class_map_path, img_size=128, conf_thresh=0.6, debounce_frames=4, pad=20, cam_id=0):
    # load
    model = tf.keras.models.load_model(model_path)
    class_names = load_class_names(class_map_path)
    action_mapper = ActionMapper()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    prev_label = None
    same_count = 0

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        fps_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            results = hands.process(frame_rgb)
            display_text = ''
            confidence = 0.0
            label = None

            if results.multi_hand_landmarks:
                # use the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(xs) * w), int(max(xs) * w)
                y_min, y_max = int(min(ys) * h), int(max(ys) * h)
                x1, y1, x2, y2 = center_pad_bbox(x_min, y_min, x_max, y_max, pad, w, h)
                crop = frame[y1:y2, x1:x2]
                if crop.size != 0 and (x2 - x1) > 10 and (y2 - y1) > 10:
                    inp = preprocess_image_for_model(crop, img_size)
                    preds = model.predict(np.expand_dims(inp, axis=0))[0]
                    idx = int(np.argmax(preds))
                    confidence = float(preds[idx])
                    if confidence >= conf_thresh:
                        label = class_names[idx]
                    else:
                        label = None

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Debounce logic: require same label for `debounce_frames` frames
            if label is None:
                same_count = 0
                prev_label = None
            else:
                if label == prev_label:
                    same_count += 1
                else:
                    same_count = 1
                prev_label = label

            if same_count >= debounce_frames:
                # trigger action and reset counter for that label (ActionMapper handles cooldown)
                action_mapper.trigger(label)
                same_count = 0
                prev_label = None

            # overlay info + FPS
            fps = 1.0 / (time.time() - fps_time + 1e-8)
            fps_time = time.time()
            info = f"Pred: {label} ({confidence:.2f})" if label else "Pred: -"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            cv2.imshow("Hand Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/final_model.h5")
    parser.add_argument("--class_map", type=str, default="models/class_indices.json")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--conf_thresh", type=float, default=0.6)
    parser.add_argument("--debounce_frames", type=int, default=4)
    parser.add_argument("--pad", type=int, default=20)
    parser.add_argument("--cam_id", type=int, default=0)
    args = parser.parse_args()
    main(args.model, args.class_map, args.img_size, args.conf_thresh, args.debounce_frames, args.pad, args.cam_id)
