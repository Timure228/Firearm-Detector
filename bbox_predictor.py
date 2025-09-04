import keras
from process_files import X_test, y_test
import numpy as np
import cv2

# Predict
idx = 11
model = keras.models.load_model("models/custom_vgg16_bbox_reg.keras")
y_pred = model.predict(np.expand_dims(X_test[idx], 0))[0] * 256
top, left, width, height = y_pred
x1, y1 = int(left), int(top)
x2, y2 = int(left) + int(width), int(top) + int(height)

print(y_pred, y_test[idx])
# Visualize with cv2
img_arr = X_test[idx].numpy()
img_arr_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
# Create a bounding box
cv2.rectangle(img_arr_rgb, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
cv2.imshow("Output", img_arr_rgb)
cv2.waitKey(0)
