import keras
from process_files import X_train, y_train, X_val, y_val

# Import the model
base_model = keras.applications.MobileNetV2(input_shape=(256, 256, 3),
                                            include_top=False,
                                            weights="imagenet")
# Define custom layers
input_layer = keras.layers.Input((256, 256, 3))
x = base_model(input_layer, training=False)
glob_avg_pool = keras.layers.GlobalAvgPool2D()(x)
x = keras.layers.Flatten()(glob_avg_pool)
x = keras.layers.Dense(256, activation="relu")(x)
dropout_layer = keras.layers.Dropout(0.5)(x)
output_layer = keras.layers.Dense(4, activation="linear")(dropout_layer)
# Define the model
custom_model = keras.Model(input_layer, output_layer, name="custom_model")
# 1 Training Loop
for layer in base_model.layers:
    layer.trainable = False
# Compile the model
loss = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.7)
custom_model.compile(loss=loss, optimizer=optimizer)
history = custom_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5)
# 2 Training Loop
for layer in base_model.layers[:20]:
    layer.trainable = False
# Compile the model
loss = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.7)
custom_model.compile(loss=loss, optimizer=optimizer)
history = custom_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15)
# 3 Training Loop
for layer in base_model.layers:
    layer.trainable = True
# Compile the model
loss = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.7)
custom_model.compile(loss=loss, optimizer=optimizer)
history = custom_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25)
# Save the model
custom_model.save("models/custom1.keras")
