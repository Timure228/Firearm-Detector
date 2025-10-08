import keras
from pathlib import Path
from dataset_tf import build_dataset
from models.yolo import MyYolo

test_images_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/datasets/test/images")

test_labels_path = Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/datasets/test/labels")

model = keras.models.load_model("saved_models/yolo_bbox_reg.keras")

test_ds = build_dataset(test_images_path, test_labels_path, batch_size=8)

print(model.predict(test_ds))

