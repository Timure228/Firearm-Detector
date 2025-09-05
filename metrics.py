import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class MeanIou(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_iou = self.add_weight(name="total_iou", initializer="zeros")  # Total score
        self.count = self.add_weight(name="count", initializer="zeros")  # Total boxes

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to x1, y1, x2, y2
        true_x1 = y_true[:, 1]
        true_y1 = y_true[:, 0]
        true_x2 = y_true[:, 2] + true_x1
        true_y2 = y_true[:, 3] + true_y1

        pred_x1 = y_pred[:, 1]
        pred_y1 = y_pred[:, 0]
        pred_x2 = y_pred[:, 2] + pred_x1
        pred_y2 = y_pred[:, 3] + pred_y1

        # Intersection
        inter_x1 = tf.maximum(true_x1, pred_x1)
        inter_y1 = tf.maximum(true_y1, pred_y1)
        inter_x2 = tf.maximum(true_x2, pred_x2)
        inter_y2 = tf.maximum(true_y2, pred_y2)

        inter_area = tf.maximum(inter_x2 - inter_x1, 0) * tf.maximum(inter_y2 - inter_y1, 0)

        # Union
        true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        union_area = true_area + pred_area - inter_area

        # Calculate Intersection over Union
        iou = inter_area / (union_area + 1e-7)

        # Accumulate total_iou (score) and count (n of bboxes)
        self.total_iou.assign_add(tf.reduce_sum(iou))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))  # convert batch_size to tf.float32

    def result(self):
        """
        Return the mean IoU of all processed bboxes
        """
        return self.total_iou / self.count

    def reset_state(self):
        """
        Reset both total_iou and count metrics
        """
        self.total_iou.assign(0.0)
        self.count.assign(0.0)
