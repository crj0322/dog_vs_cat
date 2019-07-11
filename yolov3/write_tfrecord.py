import os
import tensorflow as tf
import numpy as np
import cv2 as cv
from tqdm import tqdm

from utils import read_anchors
from data_gen import read_boxes, shuffle_split

'''write tfrecord'''

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_list_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def dense_to_sparse_value(dense):
    idx = np.nonzero(dense)
    values = dense[idx]
    idx = np.concatenate([idx], axis=1).transpose()
    sparse = tf.sparse.SparseTensor(idx, values, dense.shape)
    sparse = tf.io.serialize_sparse(sparse)
    return sparse

def create_tf_example(image_file, dst_wh, xml_file, class_dict, anchors):
    image = cv.imread(image_file)
    raw_h, raw_w = image.shape[:2]
    image = cv.resize(image, dst_wh)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    encoded_jpg = cv.imencode('.jpg', image)[1].tostring()
    filename = image_file.split(os.sep)[-1]
    label = 0 if filename[0].isupper() else 1

    y_true = read_boxes(xml_file, class_dict, anchors)

    y_sparse = []
    for y in y_true:
        y = np.squeeze(y, axis=0)
        sst = dense_to_sparse_value(y)
        y_sparse.append(sst.numpy())
    
    feature_dict = {'image': _bytes_feature(encoded_jpg),
        'raw_h': _int64_feature(raw_h),
        'raw_w': _int64_feature(raw_w),
        'class': _int64_feature(label)}

    for i in range(len(y_sparse)):
        feature_dict['y%d' % i] = _bytes_list_feature(y_sparse[i])
    
    tf_example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))
    return tf_example


def generate_tfrecord(file_list, image_path, dst_wh, xml_path, class_dict, anchors, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for filepath in tqdm(file_list):
            filename = filepath.split('.')[0]
            image_file = os.path.join(image_path, filename + '.jpg')
            xml_file = os.path.join(xml_path, filename + '.xml')
            tf_example = create_tf_example(image_file, dst_wh, xml_file, class_dict, anchors)
            writer.write(tf_example.SerializeToString())
    

def main():
    # write
    image_path =  'E:/data/The Oxford-IIIT Pet Dataset/images'
    xml_path = 'E:/data/The Oxford-IIIT Pet Dataset/annotations/xmls'
    train_path = './data/train_tiny.record'
    test_path = './data/test_tiny.record'
    ancher_path = './model/tiny_pet_anchors.txt'
    classes_name = {'cat': 0, 'dog': 1}
    dst_wh = (416, 416)
    anchors = read_anchors(ancher_path)
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
    
    generate_tfrecord(train_list, image_path, dst_wh, xml_path,
        classes_name, anchors, train_path)

    generate_tfrecord(test_list, image_path, dst_wh, xml_path,
        classes_name, anchors, test_path)
    
    
if __name__ == '__main__':
    main()
