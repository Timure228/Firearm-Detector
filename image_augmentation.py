from pathlib import Path
import keras
from process_files import images_amount, images_path
import os


def augment_imgs(img_path, img_amount, save_path) -> None:
    # Define the layers
    layers = [keras.layers.RandomFlip(mode="horizontal"),
              keras.layers.RandomRotation(factor=0.3),
              keras.layers.RandomContrast(factor=0.2)]
    # Loop through each image
    for img_idx, img_name in enumerate(sorted(os.listdir(img_path))):
        # Define the pipeline
        pipeline = keras.Sequential(layers, name="Augmentation_Pipeline")
        # Load the image
        image = keras.utils.load_img(os.path.join(img_path, img_name))
        img_array = keras.utils.img_to_array(image) / 255
        # Go through the pipeline
        augmented_img_array = pipeline(img_array)
        # Convert the augmented image array back to image and save it
        augmented_img = keras.utils.array_to_img(augmented_img_array)
        augmented_img.save(os.path.join(save_path, str(img_amount + img_idx) + ".jpg"))


if __name__ == "__main__":
    augment_imgs(images_path, images_amount, Path("C:/Users/tymur.arduch/Desktop/data/gun_holding/aug"))
