import os
import pathlib
from PIL import Image
import json
# import tensorflow as tf

images_path = pathlib.Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/images")
images_amount = len(os.listdir(images_path))
coordinates_path = pathlib.Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/labels/sorted_bbox.json")


def assign_files(path, amount, datatype) -> None:
    files = sorted(os.listdir(path))
    for file_idx, filename in enumerate(files):
        file_path = pathlib.Path(path, filename)
        new_path = pathlib.Path(path, f"{file_idx}{datatype}")
        os.rename(file_path, new_path)


def resize_images(path, width, height):
    for image in os.listdir(path):
        pil_image = Image.open(pathlib.Path(path, image))
        resized_image = pil_image.resize((width, height)).convert("RGB")
        resized_image.save(pathlib.Path(path, image))

def create_dataset(images_path, coordinates_path):
    # Load images
    tensors_list = []
    for image_name in os.listdir(images_path):
        image = Image.open(os.path.join(images_path, image_name)).convert("RGB")
        tensors_list.append(tf.convert_to_tensor(image, dtype=tf.float32) / 255)

    # Load coordinates
    bbox_coordinates = []
    with open(coordinates_path, 'r') as f:
        json_coordinates = json.loads(f.read())
    # print(json_coordinates["dataset"]["images"]["image"][14])
    for i in range(images_amount):
        bbox_coordinates.append(
            list(map(int, list(json_coordinates["dataset"]["images"]["image"][i]["box"].values())[1:])))

    return tf.stack(tensors_list), tf.convert_to_tensor(bbox_coordinates, dtype=tf.float32)


assign_files(path=images_path, amount=images_amount, datatype=".jpg")
print(images_amount)
print(os.listdir(images_path))
# resize_images(images_path, 256, 256)

# X, y split
# X, y = create_dataset(images_path, coordinates_path)
# train, val split
# X_train, y_train = X[:20], y[:20]
# X_val, y_val = X[20:], y[20:]
