import cv2 as cv
import matplotlib.pyplot as plt
import colorsys
import random
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os

def read_anchors(filepath):
    with open(filepath) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    return anchors

def box_iou(box1, box2):
    """
    Arguments:
        box1/box2: xy cordinates of left-top and right-bottom corners, shape(?, 4)
    """
    w1 = box1[:, 2] - box1[:, 0]
    h1 = box1[:, 3] - box1[:, 1]
    w2 = box2[:, 2] - box2[:, 0]
    h2 = box2[:, 3] - box2[:, 1]
    cross_w = np.maximum(0, np.minimum(box1[:, 2], box2[:, 2]) - np.maximum(box1[:, 0], box2[:, 0]))
    cross_h = np.maximum(0, np.minimum(box1[:, 3], box2[:, 3]) - np.maximum(box1[:, 1], box2[:, 1]))
    return cross_w*cross_h/(w1*h1 + w2*h2 - cross_w*cross_h)

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    image_w, image_h = image.shape[:2]
    w, h = size
    new_w = int(image_w * min(w/image_w, h/image_h))
    new_h = int(image_h * min(w/image_w, h/image_h))
    resized_image = cv.resize(image, (new_w,new_h))

    pad_left = (w-new_w)//2
    pad_top = (h-new_h)//2
    pad_right = w - new_w - pad_left
    pad_bottom = h - new_h - pad_top
    boxed_image = cv.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right,
        cv.BORDER_CONSTANT, (128,128,128))
    
    return boxed_image

def draw_bbox(image, class_names, out_boxes, out_scores, out_classes):
    thickness = (image.shape[0] + image.shape[1]) // 300
    
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        # draw box
        left, top, right, bottom = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.shape[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.shape[0], np.floor(right + 0.5).astype('int32'))
        # print(label, (left, top), (right, bottom))
        cv.rectangle(image, (left, top), (right, bottom), colors[c], thickness=thickness)
        
        # draw label
        label = '{} {:.2f}'.format(predicted_class, score)
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        if top - label_size[1] >= 0:
            cv.rectangle(image, (left, top - label_size[1]), \
                (left + label_size[0], top + base_line), colors[c], thickness=cv.FILLED)
            cv.putText(image, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
        else:
            text_origin = np.array([left, top + 1])
            cv.rectangle(image, (left, top + 1), \
                (left + label_size[0], top + base_line + label_size[1] + 1), colors[c], thickness=cv.FILLED)
            cv.putText(image, label, (left, top + label_size[1] + 1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        
    plt.figure(figsize=(10,8))
    plt.imshow(image)
    
def read_image(file_list, image_path):
    """
    Arguments:
        file_list: xml file names.
        image_path: image file path.
    """
    X = []
    for i, file in enumerate(tqdm(file_list)):
        image = cv.imread(image_path + file[:-4] + '.jpg')
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # image = letterbox_image(image, (416, 416))
        image = cv.resize(image, (416, 416))
        image = np.array(image, dtype='uint8')
        image = np.expand_dims(image, 0)
        X.append(image)
    
    X = np.concatenate(X, axis=0).astype(np.float32)
    return X/255.

def read_boxes(file_list, xml_path, class_dict, anchors):
    """
    Arguments:
        file_list: xml file names.
        xml_path: xml file path.
    Return:
        list of true boxes (xymin xymax format) in each image.
    """
    y1 = []
    y2 = []
    y3 = []
    for file in tqdm(file_list):
        tree =  ET.ElementTree(file=xml_path + os.sep + file)
        
        # image size
        w = []
        h = []
        for elem in tree.iterfind('size/width'):
            w.append(int(elem.text))
        for elem in tree.iterfind('size/height'):
            h.append(int(elem.text))

        assert len(w) == 1 and len(h) == 1
        image_wh = np.array([*w, *h])
        
        xmin = []
        xmax = []
        ymin = []
        ymax = []
        box_class = []
        for elem in tree.iterfind('object/bndbox/xmin'):
            xmin.append(int(elem.text))

        for elem in tree.iterfind('object/bndbox/xmax'):
            xmax.append(int(elem.text))

        for elem in tree.iterfind('object/bndbox/ymin'):
            ymin.append(int(elem.text))

        for elem in tree.iterfind('object/bndbox/ymax'):
            ymax.append(int(elem.text))

        for elem in tree.iterfind('object/name'):
            assert elem.text in class_dict.keys()
            box_class.append(class_dict[elem.text])

        xmin = np.array(xmin).reshape((-1, 1))
        xmax = np.array(xmax).reshape((-1, 1))
        ymin = np.array(ymin).reshape((-1, 1))
        ymax = np.array(ymax).reshape((-1, 1))
        box_class = np.array(box_class).reshape((-1, 1))

        box = np.concatenate([xmin, ymin, xmax, ymax, box_class], axis=-1)
        
        # convert to y format
        y_true = boxes_to_y(box, anchors, len(class_dict), image_wh)
        y1.append(np.expand_dims(y_true[0], 0))
        y2.append(np.expand_dims(y_true[1], 0))
        y3.append(np.expand_dims(y_true[2], 0))

    y1 = np.concatenate(y1, axis=0)
    y2 = np.concatenate(y2, axis=0)
    y3 = np.concatenate(y3, axis=0)
        
    return [y1, y2, y3]

def boxes_to_y(true_boxes, anchors, num_classes, image_wh):
    """
    transfer true boxes to yolo y format.
    Arguments:
        true_boxes: bbox absolute value in image_wh of one image,
                    value as (xmin, ymin, xmax, ymax, class), shape(?, 5).
        anchors: anchor boxe size array, shape(num_anchors, 2).
        num_classes: total class num.
        image_wh: true input image size of (w, h).
        
    Returns:
        y_true: list of yolo feature map fomat,
                shape(grid w, grid h, num_anchors, 5+num_classes),
                box xywh info normalize to (0, 1).
    """
    num_anchors = anchors.shape[0] // 3
    box_class = true_boxes[:, 4].astype(np.int32)
    xymin, xymax = true_boxes[:, 0:2], true_boxes[:, 2:4]
    
    input_size = np.array([416, 416])
    
    # calculate box center xy and wh, range(0, 416).
    boxes_wh = xymax - xymin
    boxes_xy = xymin + boxes_wh//2
    
    # normalize to range(0, 1)
    boxes_xy = boxes_xy/image_wh
    
    # grid shape
    grid_wh = [input_size//32, input_size//16, input_size//8]  # [[13, 13], [26, 26], [52, 52]]
    grid_boxes_xy = [boxes_xy * grid_wh[i] for i in range(3)]  # to grid scale, range(0, grid_wh).
    grid_index = [np.floor(grid_boxes_xy[i]) for i in range(3)]
    # boxes_xy = [(boxes_xy[i] - grid_index[i]) for i in range(3)]  # size respect to one grid, range(0, 1).
    
    # true size of xy min max cordinates relative to grid left top corner.
    anchor_xymax = anchors/2
    anchor_xymin = -anchor_xymax
    box_xymax = boxes_wh/2
    box_xymin = -box_xymax
    
    # create y_true.
    y_true = [np.zeros((grid_wh[i][1], grid_wh[i][0], num_anchors, 5+num_classes), dtype='float32') for i in range(3)]
    
    # iterate on each box
    num_boxes = true_boxes.shape[0]
    for box_index in range(num_boxes):
        # calculate iou.
        box1 = np.concatenate([box_xymin[box_index], box_xymax[box_index]]).reshape(1, -1)
        box2 = np.concatenate([anchor_xymin, anchor_xymax], axis=-1)
        iou = box_iou(box1, box2)
        
        # select the best anchor
        anchor_index = np.argmax(iou)
        layer_index = 2 - anchor_index//3
        layer_anchor_index = anchor_index % 3
        
        box_xy = boxes_xy[box_index]  # shape(2,)
        # box_wh = boxes_wh[box_index]/anchors[anchor_index]  # shape(2,)
        box_wh = boxes_wh[box_index]/image_wh  # shape(2,)， range(0, 1)
        
        #  fill in y_true.
        w = grid_index[layer_index][box_index, 0].astype('int32')
        h = grid_index[layer_index][box_index, 1].astype('int32')
        y_true[layer_index][h, w, layer_anchor_index, :2] = box_xy
        y_true[layer_index][h, w, layer_anchor_index, 2:4] = box_wh
        y_true[layer_index][h, w, layer_anchor_index, 4:5] = 1
        y_true[layer_index][h, w, layer_anchor_index, 5+box_class[box_index]] = 1
        
    return y_true