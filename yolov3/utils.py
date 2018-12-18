import cv2 as cv
import matplotlib.pyplot as plt
import colorsys
import random
import numpy as np


def read_anchors(filepath):
    with open(filepath) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    return anchors

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