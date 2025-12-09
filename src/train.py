"""
train.py

Train the model with the processed LeapGestRecog dataset.

Saves:
- models/best_model.h5   (best by validation accuracy)
- models/final_model.h5  (final model after training)
- models/class_indices.json  (mapping class->index)
"""

import os
import json
import argparse
import tensorflow as tf
from data_loader import make_generators
from model import build_mobilenetv2, build_small_cnn


def main(data_dir, out_dir, img_size=128, batch_size=32, epochs=25, use_tiny=False):
    os.makedirs(out_dir, exist_ok=True)

    # === Load data generators ===
    train_gen, val_gen, test_gen, class_names = make_generators(data_dir, img_size, batch_size)
    num_classes = len(class_names)
    print(f"\n Using data from: {data_dir}")
    print(f" Classes detected ({num_classes}): {class_names}")
    print(f" Training batches: {len(train_gen)} | Validation batches: {len(val_gen)}")

    # === Build model ===
    if use_tiny:
        print("\n Building small CNN model...")
        model = build_small_cnn(num_classes, img_size)
    else:
        print("\n Building MobileNetV2 model...")
        model = build_mobilenetv2(num_classes, img_size)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # === Callbacks ===
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(out_dir, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=6, restore_best_weights=True
    )

    # === Train ===
    print("\n🚀 Starting training...\n")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[checkpoint, reduce_lr, early_stop]
    )

    # === Save final model and metadata ===
    final_model_path = os.path.join(out_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"\n Final model saved at: {final_model_path}")

    # Save class mapping
    class_map = train_gen.class_indices
    with open(os.path.join(out_dir, 'class_indices.json'), 'w') as f:
        json.dump(class_map, f, indent=2)
    print(" Class index mapping saved.")

    print("\n Training complete. All models saved in:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed", help="processed dataset directory")
    parser.add_argument("--out_dir", type=str, default="models", help="where to save models")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--use_tiny", action='store_true', help="use small CNN instead of MobileNetV2")
    args = parser.parse_args()

    main(args.data_dir, args.out_dir, args.img_size, args.batch_size, args.epochs, args.use_tiny)
