import os
from pathlib import Path
import tensorflow as tf
import keras
from PIL import Image
import numpy as np
import math

# Labels
train_images_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/datasets/train/images")

val_images_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/datasets/valid/images")

test_images_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/datasets/test/images")

# Labels
train_labels_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/datasets/train/labels")

val_labels_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/datasets/valid/labels")

test_labels_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/datasets/test/labels")


def get_labels(path: Path, batch_size: int):
    arr = []
    files = sorted(os.listdir(path))
    n_files = len(os.listdir(path))
    n_batches = n_files // batch_size
    for b in range(n_batches):
        batch = []
        for i in files[b * batch_size:(b + 1) * batch_size]:
            with open(path.joinpath(i), "r") as f:
                batch.append(list(map(float, f.read().split()[1:])))
        arr.append(batch)
    # batch = []
    # for i in files[n_batches * batch_size:]:
    #     with open(path.joinpath(i), "r") as f:
    #         batch.append(list(map(float, f.read().split()[1:])))
    # arr.append(batch)
    return tf.stack(arr)

def get_images(path, batch_size):
    arr = []
    files = sorted(os.listdir(path))
    n_files = len(os.listdir(path))
    n_batches = n_files // batch_size
    for b in range(n_batches):
        batch = []
        for i in files[b * batch_size:(b + 1) * batch_size]:
            image = np.array(Image.open(os.path.join(path, i)).convert("RGB"), dtype=np.float32)
            batch.append(image / 255.)
        arr.append(batch)
    # batch = []
    # for i in files[n_batches * batch_size:]:
    #     image = np.array(Image.open(os.path.join(path, i)).convert("RGB"), dtype=np.float32)
    #     batch.append(image / 255.)
    # arr.append(batch)
    return tf.stack(arr)


def build_dataset(images_path, labels_path, batch_size):
    return tf.data.Dataset.from_tensor_slices((get_images(images_path, batch_size), get_labels(labels_path, batch_size)))

train_ds = build_dataset(train_images_path, train_labels_path, batch_size=8)
val_ds = build_dataset(val_images_path, val_labels_path, batch_size=8)
