"""
evaluate.py

Evaluates the saved model on the test set and prints / saves a confusion matrix and classification report.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from data_loader import make_generators

def evaluate(model_path, class_map_path, data_dir, img_size=128, batch_size=32, out_dir='reports'):
    os.makedirs(out_dir, exist_ok=True)
    # load class mapping
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)
    # invert mapping to get index->class name
    idx2class = {v:k for k,v in class_map.items()}

    _, _, test_gen, _ = make_generators(data_dir, img_size, batch_size)
    model = tf.keras.models.load_model(model_path)

    # get predictions
    y_true = test_gen.classes
    steps = int(np.ceil(test_gen.samples / test_gen.batch_size))
    preds = model.predict(test_gen, steps=steps)
    y_pred = np.argmax(preds, axis=1)

    # classification report
    target_names = [idx2class[i] for i in range(len(idx2class))]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    print("Saved reports to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/final_model.h5")
    parser.add_argument("--class_map", type=str, default="models/class_indices.json")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    evaluate(args.model_path, args.class_map, args.data_dir, args.img_size, args.batch_size)


    
