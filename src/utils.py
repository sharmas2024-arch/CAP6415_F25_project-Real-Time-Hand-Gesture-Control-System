"""
utils.py

Utility helpers: image preprocessing for inference, loading class names, simple timing tools.
"""

import json
import numpy as np
import cv2
import os

def preprocess_image_for_model(img_bgr, img_size=128):
    """
    Convert OpenCV BGR image -> model input: RGB, resized, normalized (0-1)
    img_bgr: numpy array (H,W,3)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_norm = img_resized.astype('float32') / 255.0
    return img_norm

def load_class_names(class_map_path):
    """
    Loads class mapping json and returns list of class names ordered by index.
    """
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)
    # class_map: {class_name: index}
    idx2class = {v:k for k,v in class_map.items()}
    # ensure ordering by index
    class_names = [idx2class[i] for i in range(len(idx2class))]
    return class_names

def center_pad_bbox(x1, y1, x2, y2, pad, frame_w, frame_h):
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(frame_w, x2 + pad)
    y2p = min(frame_h, y2 + pad)
    return x1p, y1p, x2p, y2p
