import time
import cv2 as cv
import numpy as np
from yolo_model import YoloV3
from utils import read_names, draw_bbox


class_name = read_names('model/coco_names.txt')
yolov3 = YoloV3()
yolov3.model.load_weights('model/yolo.h5')
orgimg = cv.imread('Fallout4-1024x576.jpg')
orgimg = cv.resize(orgimg, (416, 416))
img = cv.cvtColor(orgimg, cv.COLOR_BGR2RGB)
img = img.astype(np.float32)
img /= 255.
start = time.time()
boxes, scores, classes = yolov3.predict_img(img)
end = time.time()
print('spent time: %.3fs' % (end - start))
draw_bbox(orgimg, class_name, boxes, scores, classes)
cv.imshow('img', orgimg)
cv.waitKey()