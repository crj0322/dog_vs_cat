import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, Lambda
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
import numpy as np


def conv_unit(x, filters, kernels, padding='same', strides=1):
    x = Conv2D(filters, kernels,
               padding=padding,
               strides=strides,
               use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    return x

def residual_block(inputs, filters, n):
    x = ZeroPadding2D(((1, 0),(1, 0)))(inputs)
    x = conv_unit(x, filters, (3, 3), padding='valid', strides=2)
    for i in range(n):
        y = conv_unit(x, filters//2, (1, 1))
        y = conv_unit(y, filters, (3, 3))
        x = Add()([x, y])

    return x

def darknet53(inputs):
    x = conv_unit(inputs, 32, (3, 3))
    x = residual_block(x, 64, 1)
    x = residual_block(x, 128, 2)
    x = x_36 = residual_block(x, 256, 8)
    x = x_61 = residual_block(x, 512, 8)
    x = residual_block(x, 1024, 4)
    
    return x_36, x_61, x

def top_conv(x, filters, out_dim):
    x = conv_unit(x, filters//2, (1, 1))
    x = conv_unit(x, filters, (3, 3))
    x = conv_unit(x, filters//2, (1, 1))
    x = conv_unit(x, filters, (3, 3))
    x = conv_unit(x, filters//2, (1, 1))
    
    y = conv_unit(x, filters, (3, 3))
    y = Conv2D(out_dim, (1, 1),
               padding='same',
               kernel_regularizer=l2(5e-4))(y)
    
    return x, y

def yolo_v3(input_shape, num_anchors, num_classes):
    inputs = Input(input_shape)
    x_36, x_61, x = darknet53(inputs)
    x, y1 = top_conv(x, 1024, num_anchors*(num_classes+5))
    
    x = conv_unit(x, 256, (1, 1))
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, x_61])
    x, y2 = top_conv(x, 512, num_anchors*(num_classes+5))
    
    x = conv_unit(x, 128, (1, 1))
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, x_36])
    x, y3 = top_conv(x, 256, num_anchors*(num_classes+5))
    
    return Model(inputs, [y1, y2, y3])

def feat_to_boxes(feats, anchors, num_classes,
    max_output_size_per_class, max_output_size, iou_threshold, score_threshold):
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    num_anchors = len(anchor_mask)
    input_wh = tf.constant([416, 416], dtype=feats[0].dtype)
    batch_size = tf.shape(feats[0])[0]

    out_boxes = []
    out_scores = []

    # iterate along 3 layers.
    for i in range(3):
        # get anchor tensor
        anchors_tensor = tf.reshape(tf.constant(anchors[anchor_mask[i]], dtype=feats[0].dtype), 
            [1, 1, 1, num_anchors, 2])
        
        # get grid of shape(h, w, anchor_num, 2)
        grid_hw = tf.shape(feats[i])[1:3]
        grid_y = tf.tile(tf.reshape(tf.range(grid_hw[0]), [-1, 1, 1, 1]), [1, grid_hw[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_hw[1]), [1, -1, 1, 1]), [grid_hw[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.dtypes.cast(grid, feats[i].dtype)
        
        # get prediction values in range(0, 1)
        feature = tf.reshape(feats[i], [-1, grid_hw[0], grid_hw[1], num_anchors, num_classes + 5])
        box_xy = tf.math.sigmoid(feature[..., :2])
        box_wh = tf.math.exp(feature[..., 2:4])
        box_confidence = tf.math.sigmoid(feature[..., 4:5])
        box_class_probs = tf.math.sigmoid(feature[..., 5:])
        box_xy = (box_xy + grid) / tf.dtypes.cast(grid_hw[::-1], feature.dtype)
        box_wh = box_wh * anchors_tensor / input_wh

        # for nms
        box_minmax = tf.concat([box_xy - box_wh / 2, box_xy + box_wh / 2], axis=-1)
        box_scores = box_confidence * box_class_probs
        
        box_minmax = tf.reshape(box_minmax, [batch_size, -1, 1, 4])
        box_scores = tf.reshape(box_scores, [batch_size, -1, num_classes])
        out_boxes.append(box_minmax)
        out_scores.append(box_scores)

    # shape(batch, boxes_num, 4)
    out_boxes = tf.concat(out_boxes, axis=1)
    out_scores = tf.concat(out_scores, axis=1)

    # nms
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        out_boxes, out_scores,
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )

    return boxes, scores, classes, valid_detections

# @tf.function
def predict(img, yolo_model, anchors, num_classes, max_output_size_per_class=20,
        max_output_size=100, iou_threshold=0.5, score_threshold=0.5):
    img = tf.expand_dims(img, axis=0)
    feats = yolo_model(img)
    boxes, scores, classes, valid_detections = \
        feat_to_boxes(feats, anchors, num_classes, 
        max_output_size_per_class, max_output_size, iou_threshold, score_threshold)

    box_num = valid_detections[0]
    boxes = tf.squeeze(boxes, axis=0)[:box_num]
    scores = tf.squeeze(scores, axis=0)[:box_num]
    classes = tf.squeeze(classes, axis=0)[:box_num]
    classes = tf.dtypes.cast(classes, 'int32')
    return boxes, scores, classes

def yolo_loss(inputs, anchors, num_classes, ignore_threshold=0.5):
    y_pred = inputs[:3]
    y_true = inputs[3:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    num_anchors = len(anchor_mask)
    input_wh = tf.constant([416, 416], y_pred[0].dtype)
    batch_size = tf.shape(y_pred[0])[0]

    loss = 0
    xy_loss = 0
    wh_loss = 0
    iou_loss = 0
    bg_loss = 0
    class_loss = 0
    
    # iterate along 3 layers.
    for i in range(3):
        # get anchor tensor
        anchors_tensor = tf.reshape(tf.constant(anchors[anchor_mask[i]], dtype=y_pred[0].dtype), 
            [1, 1, 1, num_anchors, 2])
        
        # get grid of shape(h, w, anchor_num, 2)
        grid_hw = tf.shape(y_pred[i])[1:3]
        grid_y = tf.tile(tf.reshape(tf.range(grid_hw[0]), [-1, 1, 1, 1]), [1, grid_hw[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_hw[1]), [1, -1, 1, 1]), [grid_hw[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.dtypes.cast(grid, y_pred[i].dtype)
        
        # get prediction values
        feature = tf.reshape(y_pred[i], [-1, grid_hw[0], grid_hw[1], num_anchors, num_classes + 5])
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
        ture_xy = y_true[i][..., :2]
        true_wh = y_true[i][..., 2:4]
        true_confidence = y_true[i][..., 4:5]
        true_class_probs = y_true[i][..., 5:]
        raw_true_xy = ture_xy * grid_wh - grid
        raw_true_wh = tf.math.log(y_true[i][..., 2:4] / anchors_tensor * input_wh)
        raw_true_wh = tf.where(true_confidence, raw_true_wh, tf.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        
        # calculate loss

        # scale cordinate loss
        lambda_cord = 2 - y_true[i][...,2:3]*y_true[i][...,3:4]
        
        # shape(batch, grid_h, grid_w, num_anchors, 2)
        tmp_xy_loss = lambda_cord * true_confidence * tf.math.square(raw_box_xy - raw_true_xy)
        tmp_wh_loss = 0.5 * lambda_cord * true_confidence * tf.math.square(raw_box_wh - raw_true_wh)
        
        # bg loss ignore iou more than iou_threshold
        # Find ignore mask, iterate over each of batch
        pred_box = tf.concat((box_xy, box_wh), axis=-1)
        ignore_mask = tf.TensorArray(y_true[0].dtype, size=1, dynamic_size=True)
        object_mask_bool = tf.dtypes.cast(true_confidence, 'bool')
        for j in tf.range(batch_size):
            true_box = tf.boolean_mask(y_true[i][j,...,0:4], object_mask_bool[j,...,0])
            iou = box_iou(pred_box[j], true_box)
            best_iou = tf.math.reduce_max(iou, axis=-1)
            ignore_mask = ignore_mask.write(j, tf.dtypes.cast(best_iou<ignore_threshold, true_box.dtypes))

        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)
        
        iou_crossentropy = tf.losses.binary_crossentropy(true_confidence, raw_box_confidence, from_logits=True)
        tmp_iou_loss = true_confidence * iou_crossentropy
        tmp_bg_loss = ignore_mask * (1-true_confidence) * iou_crossentropy
        
        tmp_class_loss = true_confidence * tf.losses.binary_crossentropy(true_class_probs, raw_box_class_probs, from_logits=True)
        
        box_count = tf.math.maximum(tf.math.reduce_sum(true_confidence), 1)
        bg_count = tf.math.maximum(tf.math.reduce_sum(ignore_mask*(1-true_confidence)), 1)
        xy_loss += tf.math.reduce_sum(tmp_xy_loss)/box_count
        wh_loss += tf.math.reduce_sum(tmp_wh_loss)/box_count
        iou_loss += tf.math.reduce_sum(tmp_iou_loss)/box_count
        bg_loss += tf.math.reduce_sum(tmp_bg_loss)/bg_count
        class_loss += tf.math.reduce_sum(tmp_class_loss)/box_count

    loss = xy_loss + wh_loss + iou_loss + bg_loss + class_loss
    
    return loss

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
    b1 = tf.expand_dims(b1, -2)
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
    
def main():
    # build model
    model = yolo_v3((416, 416, 3), 3, 80)
    model.summary()
    model.load_weights('model/yolo.h5')
    base_model = Model(model.input, [model.layers[-4].output, model.layers[-5].output, model.layers[-6].output])
    base_model.save('model/yolo_base.h5')

if __name__ == '__main__':
    main()
