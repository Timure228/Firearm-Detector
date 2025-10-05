import keras
import tensorflow as tf
from functools import partial
from dataset_keras import train_ds, val_ds
import math

@keras.saving.register_keras_serializable()
class MyYolo(keras.models.Model):
    def __init__(self, input_shape, *args, **kwargs):
        super(MyYolo, self).__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.layers_ = self.build_layers(input_shape)

    def build_layers(self, input_shape):
        conv_layer = partial(keras.layers.Conv2D, padding="same", use_bias=False)
        max_pool = partial(keras.layers.MaxPool2D)
        # Layers
        return keras.Sequential([
            keras.layers.Input(input_shape),
            # 1 Block
            conv_layer(filters=64, kernel_size=7, strides=2),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            max_pool(pool_size=2, strides=2),
            # 2 Block
            conv_layer(filters=192, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            max_pool(pool_size=2, strides=2),
            # 3 Block
            conv_layer(filters=128, kernel_size=1),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=256, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=512, kernel_size=1),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=512, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            max_pool(pool_size=2, strides=2),
            # 4 Block
            conv_layer(filters=256, kernel_size=1),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=512, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=256, kernel_size=1),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=512, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=256, kernel_size=1),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=512, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=256, kernel_size=1),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=512, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),

            conv_layer(filters=512, kernel_size=1),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=1024, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            max_pool(pool_size=2, strides=2),
            # Block 5
            conv_layer(filters=512, kernel_size=1),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=1024, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=512, kernel_size=1),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=1024, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=1024, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=1024, kernel_size=3, strides=2),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            # Block 6
            conv_layer(filters=1024, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            conv_layer(filters=1024, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            # Dense
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(4, activation="linear")
        ])

    def get_config(self):
        config = super(MyYolo, self).get_config()
        config.update({"input_shape": self.input_shape})
        return config

    def call(self, x):
        x = self.layers_(x)
        return x

yolo = MyYolo((256, 256, 3))
# yolo.summary(expand_nested=False)

loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(0.001)
yolo.compile(loss=loss_fn, optimizer=optimizer)

history = yolo.fit(train_ds, validation_data=val_ds, epochs=5, batch_size=8)
