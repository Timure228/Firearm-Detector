import os
from pathlib import Path
import tensorflow as tf
import keras
import numpy as np
import math

# Images
train_data_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/datasets/train")

val_data_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/datasets/valid")

test_data_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/datasets/test")

# Train Set
train_X = keras.utils.image_dataset_from_directory(
    directory=train_data_path,
    follow_links=True,
    labels=None,
    color_mode="rgb",
    shuffle=False,
    image_size=(256, 256),
    batch_size=8
)
# Validation Set
val_X = keras.utils.image_dataset_from_directory(
    directory=val_data_path,
    labels=None,
    follow_links=True,
    color_mode="rgb",
    shuffle=False,
    image_size=(256, 256),
    batch_size=8
)
# Test Set
test_X = keras.utils.image_dataset_from_directory(
    directory=test_data_path,
    labels=None,
    follow_links=True,
    color_mode="rgb",
    shuffle=False,
    batch_size=8
)

# Image normalization
train_X = train_X.map(lambda x: tf.cast(x / 255., tf.float32))

val_X = val_X.map(lambda x: tf.cast(x / 255., tf.float32))

test_X = test_X.map(lambda x: tf.cast(x / 255., tf.float32))

# Labels
train_labels_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/datasets/train/labels")

val_labels_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/datasets/valid/labels")

test_labels_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/datasets/test/labels")


def get_labels(path: Path, batch_size: int):
    arr = []
    files = os.listdir(path)
    n_files = len(os.listdir(path))
    n_batches = math.floor(n_files / batch_size)
    batch_arr = []
    for i in files[:n_batches * batch_size]:
        with open(path.joinpath(i), "r") as f:
            batch_arr.append(np.array(list(map(float, f.read().split()[1:]))))
            if len(batch_arr) == 8:
                arr.append(tf.convert_to_tensor(batch_arr, dtype=tf.float32))
                batch_arr = []
    return tf.convert_to_tensor(arr, dtype=tf.float32)

# Train Set
train_y = get_labels(train_labels_path, 8)
# Validation Set
val_y = get_labels(val_labels_path, 8)
# Test Set
test_y = get_labels(test_labels_path, 8)
