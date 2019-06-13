from timeit import default_timer as timer
import cv2 as cv
import numpy as np
from yolo_model import YoloV3
from utils import read_names, read_anchors, draw_bbox, gen_colors


def detect_img(predict_func, img_path, class_name, input_size):
    orgimg = cv.imread(img_path)
    img = cv.resize(orgimg, (416, 416))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    colors = gen_colors(len(class_name))
    start = timer()
    boxes, scores, classes = predict_func(img)
    end = timer()
    print('spent time: %.3fs' % (end - start))
    draw_bbox(orgimg, class_name, boxes.numpy(), scores.numpy(), classes.numpy(), colors)
    cv.imshow('img', orgimg)
    cv.waitKey()

def detect_video(predict_func, video_path, class_name, input_size, output_path=""):
    # read video
    vid = cv.VideoCapture(video_path)
    video_size = (int(vid.get(cv.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)))
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv.CAP_PROP_FOURCC))
    video_fps = vid.get(cv.CAP_PROP_FPS)
    isOutput = True if output_path != "" else False
    if isOutput:
        out = cv.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    # fps info
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    # generate color map
    colors = gen_colors(len(class_name))

    # read frames
    while True:
        return_value, org_frame = vid.read()
        if return_value == False:
            break

        # detect
        # frame = squar_crop(frame, cropSize)
        frame = cv.resize(org_frame, input_size)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        boxes, scores, classes = predict_func(frame)
        draw_bbox(org_frame, class_name, boxes.numpy(), scores.numpy(), classes.numpy(), colors)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv.putText(org_frame, text=fps, org=(3, 15), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(0, 255, 0), thickness=2)
        cv.namedWindow("result", cv.WINDOW_NORMAL)
        cv.imshow("result", org_frame)
        if isOutput:
            out.write(org_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

# load model
class_name = read_names('model/coco_names.txt')
anchors = read_anchors('./model/yolo_anchors.txt')
yolov3 = YoloV3(input_shape=(416, 416, 3), 
        num_classes=len(class_name),
        anchors=anchors,
        training=False
        )
yolov3.model.load_weights('model/yolo.h5')

# test img
# detect_img(yolov3.predict_img, 'Fallout4-1024x576.jpg', class_name, yolov3.input_shape[:2])

# test video
video_path = 'E:\\data\\180301_16_B_LunarYearsParade_29.mp4'
detect_video(yolov3.predict_img, video_path, class_name, yolov3.input_shape[:2])
