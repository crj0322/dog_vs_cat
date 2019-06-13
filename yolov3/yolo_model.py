import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, Lambda
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
import numpy as np


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Make trainable=False freeze BN for real
    """
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def conv_unit(x, filters, kernels, training, padding='same', strides=1):
    x = Conv2D(filters, kernels,
               padding=padding,
               strides=strides,
               use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x, training=training)
    x = LeakyReLU(alpha=0.1)(x)
    
    return x

def residual_block(inputs, filters, n, training):
    x = ZeroPadding2D(((1, 0),(1, 0)))(inputs)
    x = conv_unit(x, filters, (3, 3), training, padding='valid', strides=2)
    for i in range(n):
        y = conv_unit(x, filters//2, (1, 1), training)
        y = conv_unit(y, filters, (3, 3), training)
        x = Add()([x, y])

    return x

def darknet53(inputs, training):
    x = conv_unit(inputs, 32, (3, 3), training)
    x = residual_block(x, 64, 1, training)
    x = residual_block(x, 128, 2, training)
    x = x_36 = residual_block(x, 256, 8, training)
    x = x_61 = residual_block(x, 512, 8, training)
    x = residual_block(x, 1024, 4, training)
    
    return x_36, x_61, x

def top_conv(x, filters, out_dim, training):
    x = conv_unit(x, filters//2, (1, 1), training)
    x = conv_unit(x, filters, (3, 3), training)
    x = conv_unit(x, filters//2, (1, 1), training)
    x = conv_unit(x, filters, (3, 3), training)
    x = conv_unit(x, filters//2, (1, 1), training)
    
    y = conv_unit(x, filters, (3, 3), training)
    y = Conv2D(out_dim, (1, 1),
               padding='same',
               kernel_regularizer=l2(5e-4))(y)
    
    return x, y

def yolo_v3(inputs, num_anchors, num_classes, training):
    x_36, x_61, x = darknet53(inputs, training)
    x, y1 = top_conv(x, 1024, num_anchors*(num_classes+5), training)
    
    x = conv_unit(x, 256, (1, 1), training)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, x_61])
    x, y2 = top_conv(x, 512, num_anchors*(num_classes+5), training)
    
    x = conv_unit(x, 128, (1, 1), training)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, x_36])
    x, y3 = top_conv(x, 256, num_anchors*(num_classes+5), training)
    
    return y1, y2, y3

def feat_to_boxes(feats, num_anchors, anchors, num_classes, input_wh):
    # get anchor tensor
    anchors_tensor = tf.reshape(tf.constant(anchors, dtype=feats.dtype),
        [1, 1, num_anchors, 2])
    
    # get grid of shape(h, w, 1, 2)
    grid_hw = feats.get_shape()[1:3]
    grid_x, grid_y = tf.meshgrid(tf.range(grid_hw[1]), tf.range(grid_hw[0]))
    grid = tf.expand_dims(tf.stack([grid_x, grid_y], axis=-1), axis=2)
    grid = tf.dtypes.cast(grid, feats.dtype)
    
    # get prediction values in range(0, 1)
    # batch size == 1, ignore batch dim because tflite not support tensor dim > 4
    feature = tf.reshape(feats, [grid_hw[0], grid_hw[1], num_anchors, num_classes + 5])
    box_xy = tf.math.sigmoid(feature[:, :, :, :2])
    box_wh = tf.math.exp(feature[:, :, :, 2:4])
    box_confidence = tf.math.sigmoid(feature[:, :, :, 4:5])
    box_class_probs = tf.math.sigmoid(feature[:, :, :, 5:])
    box_xy = (box_xy + grid) / tf.dtypes.cast(grid_hw[::-1], feature.dtype)
    box_wh = box_wh * anchors_tensor / input_wh

    # for nms
    half_wh = box_wh / 2
    box_minmax = tf.concat([box_xy - half_wh, box_xy + half_wh], axis=-1)
    box_scores = box_confidence * box_class_probs
    
    box_minmax = tf.reshape(box_minmax, [-1, 4])
    box_scores = tf.reshape(box_scores, [-1, num_classes])

    return box_minmax, box_scores

def layer_loss(y_true, y_pred, anchors, num_classes, input_wh, ignore_threshold):
    num_anchors = tf.shape(anchors)[0]
    batch_size = tf.shape(y_pred)[0]

    # get anchor tensor
    anchors_tensor = tf.reshape(tf.constant(anchors, dtype=y_pred.dtype),
        [1, 1, 1, num_anchors, 2])
    
    # get grid of shape(h, w, 1, 2)
    grid_hw = tf.shape(y_pred)[1:3]
    grid_x, grid_y = tf.meshgrid(tf.range(grid_hw[1]), tf.range(grid_hw[0]))
    grid = tf.expand_dims(tf.stack([grid_x, grid_y], axis=-1), axis=2)
    grid = tf.dtypes.cast(grid, y_pred.dtype)

    # get prediction values
    feature = tf.reshape(y_pred, [-1, grid_hw[0], grid_hw[1], num_anchors, num_classes + 5])
    raw_box_xy = feature[..., :2]
    raw_box_wh = feature[..., 2:4]
    raw_box_confidence = feature[..., 4:5]
    raw_box_class_probs = feature[..., 5:]
    box_xy = tf.math.sigmoid(raw_box_xy)
    box_wh = tf.math.exp(raw_box_wh)
    grid_wh = tf.dtypes.cast(grid_hw[::-1], feature.dtype)
    box_xy = (box_xy + grid) / grid_wh
    box_wh = box_wh * anchors_tensor / input_wh

    # get true values
    y_true = tf.reshape(y_true, [-1, grid_hw[0], grid_hw[1], num_anchors, num_classes + 5])
    ture_xy = y_true[..., :2]
    true_wh = y_true[..., 2:4]
    true_confidence = y_true[..., 4:5]
    true_class_probs = y_true[..., 5:]
    raw_true_xy = ture_xy * grid_wh - grid
    raw_true_wh = tf.math.log(y_true[..., 2:4] / anchors_tensor * input_wh)
    raw_true_wh = tf.where(tf.math.is_inf(raw_true_wh), tf.zeros_like(raw_true_wh), raw_true_wh)
    
    # calculate loss

    # scale cordinate loss
    lambda_cord = 2 - y_true[...,2:3] * y_true[...,3:4]
    
    # shape(batch, grid_h, grid_w, num_anchors, 2)
    xy_loss = lambda_cord * true_confidence * tf.math.square(raw_box_xy - raw_true_xy)
    wh_loss = 0.5 * lambda_cord * true_confidence * tf.math.square(raw_box_wh - raw_true_wh)
    
    # bg loss ignore iou more than iou_threshold
    # Find ignore mask, iterate over each of batch
    pred_box = tf.concat((box_xy, box_wh), axis=-1)
    ignore_mask = tf.TensorArray(y_true[0].dtype, size=1, dynamic_size=True)
    object_mask_bool = tf.dtypes.cast(true_confidence, 'bool')
    def loop_body(b, ignore_mask):
        true_box = tf.boolean_mask(y_true[b,...,0:4], object_mask_bool[b,...,0])
        iou = box_iou(pred_box[b], true_box)
        best_iou = tf.math.reduce_max(iou, axis=-1)
        ignore_mask = ignore_mask.write(b, tf.dtypes.cast(best_iou<ignore_threshold, true_box.dtype))
        return b+1, ignore_mask
    _, ignore_mask = tf.while_loop(lambda b,*args: b<batch_size, loop_body, [0, ignore_mask])

    ignore_mask = ignore_mask.stack()
    ignore_mask = tf.expand_dims(ignore_mask, -1)
    
    iou_crossentropy = tf.losses.binary_crossentropy(true_confidence, raw_box_confidence, from_logits=True)
    iou_crossentropy = tf.expand_dims(iou_crossentropy, axis=-1)
    iou_loss = true_confidence * iou_crossentropy
    bg_loss = ignore_mask * (1 - true_confidence) * iou_crossentropy
    
    class_loss = tf.losses.binary_crossentropy(true_class_probs, raw_box_class_probs, from_logits=True)
    class_loss = true_confidence * tf.expand_dims(class_loss, axis=-1)

    box_count = tf.math.maximum(tf.math.reduce_sum(true_confidence), 1)
    bg_count = tf.math.maximum(tf.math.reduce_sum(ignore_mask * (1 - true_confidence)), 1)
    xy_loss = tf.math.reduce_sum(xy_loss)/box_count
    wh_loss = tf.math.reduce_sum(wh_loss)/box_count
    iou_loss = tf.math.reduce_sum(iou_loss)/box_count
    bg_loss = tf.math.reduce_sum(bg_loss)/bg_count
    class_loss = tf.math.reduce_sum(class_loss)/box_count

    return xy_loss + wh_loss + iou_loss + bg_loss + class_loss

def box_iou(b1, b2):
    '''Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    '''

    # Expand dim to apply broadcasting.
    b1 = tf.expand_dims(b1, axis=-2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = tf.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.math.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.math.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.math.minimum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

class YoloV3():
    def __init__(self, input_shape, 
        num_classes, 
        anchors,
        training,
        num_anchors=3, 
        anchor_mask=[[6,7,8], [3,4,5], [0,1,2]]):
        # model parameters
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.anchor_mask = anchor_mask
        self.anchors = anchors

        # nms parameters
        self.max_output_size_per_class = 20
        self.max_output_size = 100
        self.iou_threshold = 0.5
        self.score_threshold = 0.3

        # training parameters
        self.ignore_threshold = 0.5
        self.training = training

        self.model = self.build_model()
        
    def build_model(self):
        inputs = Input(self.input_shape)
        y1, y2, y3 = yolo_v3(inputs, self.num_anchors, self.num_classes, self.training)
        if self.training:
            return Model(inputs, [y1, y2, y3])

        outputs = Lambda(self.get_boxes)([y1, y2, y3])
        return Model(inputs, outputs)
    
    def get_boxes(self, inputs):
        out_boxes = []
        out_scores = []

        # iterate along out layers.
        for i, feat in enumerate(inputs):
            boxes, scores = feat_to_boxes(feat, self.num_anchors,
                self.anchors[self.anchor_mask[i]], self.num_classes, self.input_shape[:2])
            out_boxes.append(boxes)
            out_scores.append(scores)
        out_boxes = tf.concat(out_boxes, axis=0)
        out_scores = tf.concat(out_scores, axis=0)

        return out_boxes, out_scores

    def yolo_loss(self, i):
        return lambda y_true, y_pred: layer_loss(y_true, y_pred, 
            self.anchors[self.anchor_mask[i]], self.num_classes, self.input_shape[:2], self.ignore_threshold)

    @tf.function
    def predict_img(self, img):
        img = tf.dtypes.cast(img, 'float32')
        img = tf.expand_dims(img, axis=0)
        img /= 255.
        out_boxes, out_scores = self.model(img)
        out_boxes = tf.reshape(out_boxes, [1, -1, 1, 4])
        out_scores = tf.reshape(out_scores, [1, -1, self.num_classes])
        boxes, scores, classes, valid_detections = \
            tf.image.combined_non_max_suppression(
            out_boxes, out_scores,
            max_output_size_per_class=self.max_output_size_per_class,
            max_total_size=self.max_output_size,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold)

        box_num = valid_detections[0]
        boxes = tf.squeeze(boxes, axis=0)[:box_num]
        scores = tf.squeeze(scores, axis=0)[:box_num]
        classes = tf.squeeze(classes, axis=0)[:box_num]
        classes = tf.dtypes.cast(classes, 'int32')
        return boxes, scores, classes

    
def main():
    # build model
    from utils import read_anchors
    anchors = read_anchors('./model/yolo_anchors.txt')
    yolov3 = YoloV3(input_shape=(416, 416, 3), 
        num_classes=80,
        anchors=anchors,
        training=False
        )
    yolov3.model.summary()
    yolov3.model.load_weights('model/yolo.h5')

    # base_model = Model(yolov3.model.input,
    #     [yolov3.model.layers[-4].output, yolov3.model.layers[-5].output, yolov3.model.layers[-6].output])
    # base_model.save('model/yolo_base.h5')

    # Convert to tflite.
    converter = tf.lite.TFLiteConverter.from_keras_model(yolov3.model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()
    open("model/yolov3.tflite","wb").write(tflite_model)

if __name__ == '__main__':
    main()
