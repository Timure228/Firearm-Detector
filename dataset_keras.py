from pathlib import Path
import keras

train_data_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/labels/train")

val_data_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/labels/valid")

test_data_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/labels/test")

# Train Set
train_ds = keras.utils.image_dataset_from_directory(
    directory=train_data_path,
    labels="inferred",
    follow_links=True,
    color_mode="rgb",
    shuffle=True,
    image_size=(256, 256)
)
# Validation Set
val_ds = keras.utils.image_dataset_from_directory(
    directory=val_data_path,
    labels="inferred",
    follow_links=True,
    color_mode="rgb",
    shuffle=True,
    image_size=(256, 256)
)
# Test Set
test_ds = keras.utils.image_dataset_from_directory(
    directory=test_data_path,
    labels="inferred",
    follow_links=True,
    color_mode="rgb",
    shuffle=True,
    batch_size=8
)