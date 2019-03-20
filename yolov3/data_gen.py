import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import random
import numpy as np
import cv2 as cv
from utils import box_iou


def read_file_list(in_file):
    assert os.path.exists(in_file)
    fo = open(in_file, 'r')
    lines = fo.readlines()
    file_list = []
    for line in lines:
        file_list.append(line.strip())
    fo.close()
    return file_list

def write_file_list(out_file, file_list):
    if os.path.exists(out_file):
        os.remove(out_file)
    fo = open(out_file, 'w')
    for file in file_list:
        fo.write(file + '\n')
    fo.close()

def shuffle_split(file_list, split_rate=0.1, seed=None):
    test_num = int(len(file_list)*split_rate)
    if seed != None:
        np.random.seed(seed)
    file_index = list(range(len(file_list)))
    np.random.shuffle(file_index)
    test_index = file_index[:test_num]
    train_index = file_index[test_num:]
    train_list = []
    test_list = []
    for i in train_index:
        train_list.append(file_list[i])
    for i in test_index:
        test_list.append(file_list[i])

    return train_list, test_list

def data_generator(file_list, img_path, xml_path, class_dict, anchors,
    batch_size=32, shuffle=True, loop=True):
    """
    Arguments:
        file_list: file names.
        img_path: full path of image dir.
        xml_path: full path of xml dir.
        class_dict: dict like {'cat': 0, 'dog': 1}.
        anchors: anchor priors.
    """
    file_index = list(range(len(file_list)))
    if shuffle:
        np.random.shuffle(file_index)

    start = 0
    while True:
        end = (start + batch_size) % len(file_index)
        if end < start:
            if loop:
                batch_index = file_index[start:] + file_index[:end]
            else:
                batch_index = file_index[start:]
                end = 0
        else:
            batch_index = file_index[start:end]
        start = end

        X = []
        y1 = []
        y2 = []
        y3 = []
        for i in batch_index:
            # generate X
            image = cv.imread(os.path.join(img_path, file_list[i] + '.jpg'))
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            # image = letterbox_image(image, (416, 416))
            image = cv.resize(image, (416, 416)).astype(np.float32)
            image /= 255.
            image = np.expand_dims(image, 0)
            X.append(image)

            # generate y
            y_true = read_boxes(os.path.join(xml_path, file_list[i] + '.xml'),
                class_dict, anchors)
            y1.append(y_true[0])
            y2.append(y_true[1])
            y3.append(y_true[2])
        
        X = np.concatenate(X, axis=0).astype(np.float32)/255.
        y1 = np.concatenate(y1, axis=0)
        y2 = np.concatenate(y2, axis=0)
        y3 = np.concatenate(y3, axis=0)
        yield [X, y1, y2, y3], np.zeros(batch_size)

def read_boxes(file_path, class_dict, anchors):
    """
    Arguments:
        file_path: full file path of xml file.
        class_dict: dict like {'cat': 0, 'dog': 1}.
        anchors: anchor priors.
    Return:
        true boxes map (normalized xywh format) in each image.
    """
    tree =  ET.ElementTree(file=file_path)
    
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
    
    return y_true

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
    y_true = [np.zeros((1, grid_wh[i][1], grid_wh[i][0], num_anchors, 5+num_classes),
        dtype='float32') for i in range(3)]
    
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
        y_true[layer_index][0, h, w, layer_anchor_index, :2] = box_xy
        y_true[layer_index][0, h, w, layer_anchor_index, 2:4] = box_wh
        y_true[layer_index][0, h, w, layer_anchor_index, 4:5] = 1
        y_true[layer_index][0, h, w, layer_anchor_index, 5+box_class[box_index]] = 1
        
    return y_true

def main():
    classes_name = {'cat': 0, 'dog': 1}
    from utils import read_anchors
    anchors = read_anchors('model/pet_anchors.txt')
    xml_path = 'E:/data/The Oxford-IIIT Pet Dataset/annotations/xmls'
    image_path = 'E:/data/The Oxford-IIIT Pet Dataset/images/'
    file_list = os.listdir(xml_path)

    # shuffle split
    cat_list = []
    dog_list = []
    for file in file_list:
        filename = file.split('.')[0]
        if file[0].isupper():
            cat_list.append(filename)
        else:
            dog_list.append(filename)
    del file_list
    data_set = [cat_list, dog_list]
    train_list = []
    test_list = []
    for file_list in data_set:
        tmp_train_list, tmp_test_list = shuffle_split(file_list)
        train_list += tmp_train_list
        test_list += tmp_test_list

    # test generator
    batch_gen = data_generator(test_list, image_path, xml_path, classes_name, anchors, loop=False)
    while True:
        batch = next(batch_gen)[0]
        print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape)
        os.system('pause')

if __name__ == '__main__':
    main()
