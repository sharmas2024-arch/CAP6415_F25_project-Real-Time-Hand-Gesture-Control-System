"""
data_loader.py

Provides convenience functions to create train/val/test generators using Keras ImageDataGenerator.
We use generators for simplicity and to leverage on-the-fly augmentation.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def make_generators(data_dir, img_size=128, batch_size=32, seed=42):
    """
    data_dir should have subfolders: train/, val/, test/, each with class folders.
    Returns: train_gen, val_gen, test_gen, class_names
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=(0.8,1.2),
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=seed
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=seed
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # class indices mapping dict
    class_names = list(train_gen.class_indices.keys())

    return train_gen, val_gen, test_gen, class_names
