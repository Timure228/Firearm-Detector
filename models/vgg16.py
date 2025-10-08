import keras
from process_files import X_train, y_train, X_val, y_val
import matplotlib.pyplot as plt
import numpy as np
from metrics import MeanIou

# Import VGG16 Model
vgg16 = keras.applications.VGG16(weights="imagenet", include_top=False,
                                 input_shape=(256, 256, 3))
# Freeze all the layers
vgg16.trainable = False

# Flatten the VGG16 max-pooling output
vgg_output = vgg16.output
flatten_layer = keras.layers.Flatten()(vgg_output)

# Fully connected layers for bounding box coordinates
bbox = keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")(flatten_layer)
bbox = keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal")(bbox)
dropout = keras.layers.Dropout(0.5)(bbox)
bbox = keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal")(dropout)
output_ = keras.layers.Dense(4, activation="linear")(bbox)

# Define the model
custom_vgg16 = keras.models.Model(vgg16.input, output_)

# 1 Train Loop
# Compile the model
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(0.01)
custom_vgg16.compile(loss=loss_fn, optimizer=optimizer, metrics=[MeanIou()])

# Fit
history = custom_vgg16.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=6, batch_size=8)

# 2 Train Loop
# Unfreeze all layers
for layer in vgg16.layers:
    layer.trainable = True

# Compile the model
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(0.001)
custom_vgg16.compile(loss=loss_fn, optimizer=optimizer, metrics=[MeanIou()])

# Fit
history = custom_vgg16.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8)

# Plot the train/val losses
custom_vgg16.save("saved_models/custom_vgg16_bbox_reg.keras")
epochs_n = 10
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs_n), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs_n), history.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.show()
