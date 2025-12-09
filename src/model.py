"""
model.py
Builds, trains, and evaluates gesture classification models.
"""

import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from visualize_results import (
    plot_history,
    plot_confusion,
    save_sample_predictions,
    save_model_architecture
)

# ------------------------------------------------------
# MODEL DEFINITIONS
# ------------------------------------------------------

def build_mobilenetv2(num_classes, img_size=128, dropout=0.3):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base.input, outputs=outputs)
    return model


def build_small_cnn(num_classes, img_size=128):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# ------------------------------------------------------
# TRAINING SCRIPT
# ------------------------------------------------------
if __name__ == "__main__":

    IMG_SIZE = 128
    BATCH = 32
    NUM_CLASSES = 10

    train_dir = "data/processed/train"
    val_dir = "data/processed/val"

    # Create output dir
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ---------------------------
    # LOAD DATA
    # ---------------------------
    datagen = ImageDataGenerator(rescale=1/255.)

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=False
    )

    # ---------------------------
    # BUILD & TRAIN MODEL
    # ---------------------------
    print("\n Building MobileNetV2 model...\n")
    model = build_mobilenetv2(NUM_CLASSES, IMG_SIZE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n Training Started...\n")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )

    print("\n Training complete.\n")

    # ---------------------------
    # SAVE RESULTS
    # ---------------------------
    plot_history(history, out_dir="results")
    plot_confusion(model, val_gen, out_dir="results")
    save_sample_predictions(model, val_gen, out_dir="results")
    save_model_architecture(model, out_dir="results")

    model.save("models/hand_gesture_mobilenet.h5")

    print("\n All results saved in /results and /models\n")
