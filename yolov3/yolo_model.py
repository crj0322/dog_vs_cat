from keras.layers import Input, Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
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
    x = residual_block(x, 256, 8)
    x = residual_block(x, 512, 8)
    x = residual_block(x, 1024, 4)
    
    return x

def top_conv(x, filters, out_dim):
    x = conv_unit(x, filters//2, (1, 1))
    x = conv_unit(x, filters, (3, 3))
    x = conv_unit(x, filters//2, (1, 1))
    x = conv_unit(x, filters, (3, 3))
    x = conv_unit(x, filters//2, (1, 1))
    
    y = conv_unit(x, filters, (3, 3))
    y = Conv2D(out_dim, (1, 1),
               padding='same',
               # kernel_initializer=init_func,
               kernel_regularizer=l2(5e-4))(y)
    
    return x, y

def yolo_model(inputs, num_anchors, num_classes):
    darknet = Model(inputs=inputs, outputs=darknet53(inputs))
    x, y1 = top_conv(darknet.output, 1024, num_anchors*(num_classes+5))
    
    x = conv_unit(x, 256, (1, 1))
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = top_conv(x, 512, num_anchors*(num_classes+5))
    
    x = conv_unit(x, 128, (1, 1))
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = top_conv(x, 256, num_anchors*(num_classes+5))
    
    return [y1, y2, y3]

def feat_to_boxes(feats, anchors, num_classes):
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    num_anchors = len(anchor_mask)
    input_wh = K.constant([416, 416])
    batch_size = K.shape(feats[0])[0]

    out_boxes = []

    # iterate along 3 layers.
    for i in range(3):
        # get anchor tensor
        anchors_tensor = K.reshape(K.constant(anchors[anchor_mask[i]]), [1, 1, 1, num_anchors, 2])
        
        # get grid
        grid_hw = K.shape(feats[i])[1:3]
        grid_y = K.tile(K.reshape(K.arange(0, stop=grid_hw[0]), [-1, 1, 1, 1]), [1, grid_hw[1], 1, 1])
        grid_x = K.tile(K.reshape(K.arange(0, stop=grid_hw[1]), [1, -1, 1, 1]), [grid_hw[0], 1, 1, 1])
        grid = K.concatenate([grid_x, grid_y])
        grid = K.cast(grid, K.dtype(feats[i]))
        
        # get prediction values in range(0, 1)
        feature = K.reshape(feats[i], [-1, grid_hw[0], grid_hw[1], num_anchors, num_classes + 5])
        box_xy = K.sigmoid(feature[..., :2])
        box_wh = K.exp(feature[..., 2:4])
        box_confidence = K.sigmoid(feature[..., 4:5])
        box_class_probs = K.sigmoid(feature[..., 5:])
        box_xy = (box_xy + grid) / K.cast(grid_hw[::-1], K.dtype(feature))
        box_wh = box_wh * anchors_tensor / K.cast(input_wh, K.dtype(feature))
        
        boxes = K.concatenate((box_xy, box_wh, box_confidence, box_class_probs))
        boxes = K.reshape(boxes, [batch_size, -1, num_classes + 5])
        out_boxes.append(boxes)

    out_boxes = K.concatenate(out_boxes, axis=1)

    return out_boxes

def yolo_loss(inputs, anchors, num_classes, ignore_threshold=0.5):
    y_pred = inputs[:3]
    y_true = inputs[3:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    num_anchors = len(anchor_mask)
    input_wh = K.constant([416, 416], K.dtype(y_pred[0]))
    batch_size = K.shape(y_pred[0])[0]
    # m = K.cast(batch_size, K.dtype(y_pred[0]))
    loss = 0
    xy_loss = 0
    wh_loss = 0
    iou_loss = 0
    bg_loss = 0
    class_loss = 0
    
    # iterate along 3 layers.
    for i in range(3):
        # get anchor tensor
        anchors_tensor = K.reshape(K.constant(anchors[anchor_mask[i]]), [1, 1, 1, num_anchors, 2])
        
        # get grid
        grid_hw = K.shape(y_pred[i])[1:3]
        grid_y = K.tile(K.reshape(K.arange(0, stop=grid_hw[0]), [-1, 1, 1, 1]), [1, grid_hw[1], 1, 1])
        grid_x = K.tile(K.reshape(K.arange(0, stop=grid_hw[1]), [1, -1, 1, 1]), [grid_hw[0], 1, 1, 1])
        grid = K.concatenate([grid_x, grid_y])
        grid = K.cast(grid, K.dtype(y_pred[i]))
        
        # get prediction values
        feature = K.reshape(y_pred[i], [-1, grid_hw[0], grid_hw[1], num_anchors, num_classes + 5])
        raw_box_xy = feature[..., :2]
        raw_box_wh = feature[..., 2:4]
        raw_box_confidence = feature[..., 4:5]
        raw_box_class_probs = feature[..., 5:]
        box_xy = K.sigmoid(raw_box_xy)
        box_wh = K.exp(raw_box_wh)
        # box_confidence = K.sigmoid(feature[..., 4:5])
        # box_class_probs = K.sigmoid(feature[..., 5:])
        grid_wh = K.cast(grid_hw[::-1], K.dtype(feature))
        box_xy = (box_xy + grid) / grid_wh
        box_wh = box_wh * anchors_tensor / input_wh
        
        # get true values
        ture_xy = y_true[i][..., :2]
        true_wh = y_true[i][..., 2:4]
        true_confidence = y_true[i][..., 4:5]
        true_class_probs = y_true[i][..., 5:]
        raw_true_xy = ture_xy * grid_wh - grid
        raw_true_wh = K.log(y_true[i][..., 2:4] / anchors[anchor_mask[i]] * input_wh)
        raw_true_wh = K.switch(true_confidence, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        
        # calculate loss

        # scale cordinate loss
        lambda_cord = 2 - y_true[i][...,2:3]*y_true[i][...,3:4]
        
        # shape(batch, grid_h, grid_w, num_anchors, 2)
        tmp_xy_loss = lambda_cord * true_confidence * K.square(raw_box_xy - raw_true_xy)
        tmp_wh_loss = 0.5 * lambda_cord * true_confidence * K.square(raw_box_wh - raw_true_wh)
        
        # bg loss ignore iou more than iou_threshold
        # Find ignore mask, iterate over each of batch.
        pred_box = K.concatenate((box_xy, box_wh))
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(true_confidence, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[i][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_threshold, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<batch_size, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)
        
        iou_crossentropy = K.binary_crossentropy(true_confidence, raw_box_confidence, from_logits=True)
        tmp_iou_loss = true_confidence * iou_crossentropy
        tmp_bg_loss = ignore_mask * (1-true_confidence) * iou_crossentropy
        
        tmp_class_loss = true_confidence * K.binary_crossentropy(true_class_probs, raw_box_class_probs, from_logits=True)
        
        box_count = K.maximum(K.sum(true_confidence), 1)
        bg_count = K.maximum(K.sum(ignore_mask*(1-true_confidence)), 1)
        xy_loss += K.sum(tmp_xy_loss)/box_count
        wh_loss += K.sum(tmp_wh_loss)/box_count
        iou_loss += K.sum(tmp_iou_loss)/box_count
        bg_loss += K.sum(tmp_bg_loss)/bg_count
        class_loss += K.sum(tmp_class_loss)/box_count

    loss = xy_loss + wh_loss + iou_loss + bg_loss + class_loss

    # for print
    # loss = K.stack([xy_loss, wh_loss, iou_loss, bg_loss, class_loss])
        
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
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou
    
def main():
    # build model
    inputs = Input((416, 416, 3))
    y1, y2, y3 = yolo_model(inputs, 3, 80)
    model = Model(inputs, [y1, y2, y3])
    model.load_weights('model/yolo.h5')
    base_model = Model(model.input, [model.layers[-4].output, model.layers[-5].output, model.layers[-6].output])
    base_model.save('model/yolo_base.h5')

if __name__ == '__main__':
    main()
