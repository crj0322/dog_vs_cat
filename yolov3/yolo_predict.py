from timeit import default_timer as timer
import cv2 as cv
import numpy as np
from yolo_model import YoloV3
from utils import read_names, read_anchors, draw_bbox


class_name = read_names('model/coco_names.txt')
anchors = read_anchors('./model/yolo_anchors.txt')
yolov3 = YoloV3(input_shape=(416, 416, 3), 
        num_classes=80,
        anchors=anchors,
        training=False
        )
yolov3.model.load_weights('model/yolo.h5')
orgimg = cv.imread('Fallout4-1024x576.jpg')
orgimg = cv.resize(orgimg, (416, 416))
img = cv.cvtColor(orgimg, cv.COLOR_BGR2RGB)
img = img.astype(np.float32)
img /= 255.
start = timer()
boxes, scores, classes = yolov3.predict_img(img)
end = timer()
print('spent time: %.3fs' % (end - start))
draw_bbox(orgimg, class_name, boxes, scores, classes)
cv.imshow('img', orgimg)
cv.waitKey()