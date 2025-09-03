import keras
from process_files import X_train, y_train, X_val, y_val

imagenet = keras.applications.Xception(input_shape=(256, 256, 3),
                                       classes=4,
                                       include_top=False)
# Edit layers
input_ = keras.layers.Input((256, 256, 3))
x = imagenet(input_, training=False)
x = keras.layers.GlobalAvgPool2D()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation="relu")(x)
dropout_layer = keras.layers.Dropout(0.5)(x)
output_ = keras.layers.Dense(4, activation="linear")(dropout_layer)
# Define the model
xception_custom = keras.models.Model(input_, output_, name="xception_custom")
# 1 Training Loop
for layer in xception_custom.layers[50:]:
    layer.trainable = False
# Compile the model
loss = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.7)
xception_custom.compile(loss=loss, optimizer=optimizer)
history = xception_custom.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5)
# 2 Training Loop
for layer in xception_custom.layers[:60]:
    layer.trainable = True
# Compile the model
loss = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.7)
xception_custom.compile(loss=loss, optimizer=optimizer)
history = xception_custom.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15)
# 3 Training Loop
for layer in xception_custom.layers:
    layer.trainable = True
# Compile the model
loss = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.7)
xception_custom.compile(loss=loss, optimizer=optimizer)
history = xception_custom.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25)
# Save the model
xception_custom.save("models/custom_xception.keras")
