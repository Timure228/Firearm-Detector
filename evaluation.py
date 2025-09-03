import keras
from process_files import X_test, X_val, y_test
import numpy as np

model = keras.models.load_model("models/custom1.keras")
print(X_test.shape, X_val.shape)
print(model.predict(np.expand_dims(X_test[0], 0)))
print(y_test[0])