import time
import cv2 as cv
import numpy as np
from yolo_model import yolo_v3, predict
from utils import read_anchors, read_names, draw_bbox


anchors = read_anchors('./model/yolo_anchors.txt')
class_name = read_names('./model/coco_names.txt')
yolo_model = yolo_v3((416, 416, 3), 3, len(class_name))
yolo_model.load_weights('model/yolo.h5')
orgimg = cv.imread('Fallout4-1024x576.jpg')
orgimg = cv.resize(orgimg, (416, 416))
img = cv.cvtColor(orgimg, cv.COLOR_BGR2RGB)
img = img.astype(np.float32)
img /= 255.
start = time.time()
boxes, scores, classes = predict(img, yolo_model, anchors, len(class_name))
end = time.time()
print('spent time: ', end - start, ' s')
draw_bbox(orgimg, class_name, boxes.numpy(), scores.numpy(), classes.numpy())
cv.imshow('img', orgimg)
cv.waitKey()