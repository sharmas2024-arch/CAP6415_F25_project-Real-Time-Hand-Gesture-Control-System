# CAP6415_F25_project-Real-Time-Hand-Gesture-Control-System
This project focuses on recognizing static hand gestures from image data using deep learning.
The objective is to enable more natural human-computer interaction (HCI) by accurately classifying various hand poses.
We utilize the LeapGestRecog dataset, which contains over 20,000 RGB images of 10 distinct hand gestures collected from multiple subjects.

Each image is preprocessed (grayscale conversion, normalization, resizing) and used to train a Convolutional Neural Network (CNN) built with TensorFlow and Keras.
Data augmentation techniques such as rotation, zoom, and horizontal flipping are applied to improve model generalization and robustness.

This work demonstrates how deep learning can be applied to build efficient gesture-based control systems for use in robotics, gaming, and virtual reality environments.
All visual results, including accuracy curves, confusion matrices, and sample predictions, are stored in the results/ folder.

1️⃣ Data Loading and Preprocessing

Loaded gesture images from the LeapGestRecog dataset.

Converted to grayscale and resized to (128 × 128) pixels.

Normalized pixel values to range [0, 1].

Augmented training data to improve diversity and avoid overfitting.

2️⃣ Model Architecture

A Convolutional Neural Network (CNN) was built with:

3 convolutional layers (ReLU activations, MaxPooling)

Fully connected dense layers with dropout

Softmax output layer for 10 gesture classes

3️⃣ Training

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Batch Size: 32

Epochs: 25–50 (depending on hardware)

Train/Validation Split: 80% / 20%

4️⃣ Evaluation

Accuracy and loss visualized using Matplotlib.


 HandGestureRecognition

 README.md                ← Project description

 requirements.txt         ← Dependencies

 data_loader.py           ← Loads and augments LeapGestRecog dataset

 vision.py                ← Main training & evaluation script

 models/                  ← Saved model weights

 results/                 ← Plots, metrics, and sample outputs



🧠 Key Learnings

CNNs are highly effective for visual pattern recognition tasks.

Data augmentation greatly improves generalization.

Gesture recognition can enable real-time HCI applications.

🧾 References

LeapGestRecog Dataset – Kaggle

Chollet, F. Deep Learning with Python, 2nd Edition

TensorFlow Documentation

KathyFeiyang cs231n Project

Joyce Nerd CV Lane Detection
