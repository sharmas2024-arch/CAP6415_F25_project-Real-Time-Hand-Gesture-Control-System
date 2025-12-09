import cv2
import tensorflow as tf
import json
import numpy as np
import os
import sys

# ------------------------
# Ensure src folder is in path to import action_mapper
# ------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from action_mappers import ActionMapper 

# ------------------------
# Load trained model
# ------------------------
model_path = os.path.join(script_dir, "../models/best_model.h5")
model = tf.keras.models.load_model(model_path)

# Load class indices
class_indices_path = os.path.join(script_dir, "../models/class_indices.json")
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# Reverse mapping: label -> gesture name
label_to_gesture = {v: k for k, v in class_indices.items()}

# ------------------------
# Initialize ActionMapper
# ------------------------
mapper = ActionMapper()

# ------------------------
# Start webcam
# ------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Press ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ------------------------
    # Preprocess frame for model
    # ------------------------
    img = cv2.resize(frame, (128, 128))        # Resize to model input (128x128)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR -> RGB
    img = img / 255.0                           # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)          # Add batch dimension

    # ------------------------
    # Predict gesture
    # ------------------------
    pred = model.predict(img)
    class_idx = int(np.argmax(pred))
    gesture = label_to_gesture[class_idx]

    # ------------------------
    # Trigger action
    # ------------------------
    mapper.trigger(gesture)

    # ------------------------
    # Display webcam feed
    # ------------------------
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Control", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
