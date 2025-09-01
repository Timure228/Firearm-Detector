import os
import pathlib
from PIL import Image
import json
import tensorflow as tf
import time

images_path = pathlib.Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/images")
images_amount = len(os.listdir(images_path))
coordinates_path = pathlib.Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/labels/json.json")


def assign_files(path, amount, datatype) -> None:
    files = sorted(os.listdir(path))
    for file_idx, filename in enumerate(files):
        file_path = pathlib.Path(path, filename)
        new_path = pathlib.Path(path, f"{file_idx}{datatype}")
        os.rename(file_path, new_path)


def resize_images(path, width, height) -> None:
    for image in os.listdir(path):
        pil_image = Image.open(pathlib.Path(path, image))
        resized_image = pil_image.resize((width, height)).convert("RGB")
        resized_image.save(pathlib.Path(path, image))


def create_dataset(images_path, coordinates_path):
    # Load images
    tensors_list = []
    for image_name in sorted(os.listdir(images_path), key=lambda x: int(x.split('.')[0])):
        image = Image.open(os.path.join(images_path, image_name)).convert("RGB")
        tensors_list.append(tf.convert_to_tensor(image, dtype=tf.float32) / 255)

    # Load coordinates
    bbox_coordinates = []
    with open(coordinates_path, 'r') as f:
        json_coordinates = json.loads(f.read())
        sorted_json = []
        # Sort coordinates
        for number in range(images_amount):
            for idx in range(images_amount):
                if int(json_coordinates["dataset"]["images"]["image"][i]["_file"].replace(".jpg", "")) == number:
                    sorted_json.append(json_coordinates["dataset"]["images"]["image"][idx])
                    break
        # Add coordinates to a list
        for idx in range(images_amount):
            bbox_coordinates.append(list(map(int, list(sorted_json[idx]["box"].values())[1:])))

    return tf.stack(tensors_list), tf.convert_to_tensor(bbox_coordinates, dtype=tf.float32)


# assign_files(path=images_path, amount=images_amount, datatype=".jpg")
# resize_images(images_path, 256, 256)

# X, y split
# X, y = create_dataset(images_path, coordinates_path)
# Train, Val, Split
# X_train, y_train = X[:(len(X)*0.75)], y[:(len(X)*0.75)]
# X_val, y_val = X[(len(X)*0.75):(len(X)*0.90)], y[(len(X)*0.75):(len(X)*0.90)]
# X_test, y_test = X[(len(X)*0.90):], y[(len(X)*0.90):]
# print(f"Train: {len(X_train)} \n"
#       f"Val: {len(X_val)} \n"
#       f"Test: {len(X_test)}")
if __name__ == "__main__":
    pass
