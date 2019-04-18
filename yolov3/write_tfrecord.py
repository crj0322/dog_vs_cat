import os
import tensorflow as tf
import numpy as np
import cv2 as cv
from tqdm import tqdm

from utils import read_anchors
from data_gen import read_boxes

flags = tf.app.flags

flags.DEFINE_string('image_path', 'E:/data/The Oxford-IIIT Pet Dataset/images', 'Path to images (directory).')
flags.DEFINE_string('xml_path', 'E:/data/The Oxford-IIIT Pet Dataset/annotations/xmls', 'Path to bounding box info.')
flags.DEFINE_string('output_path', './data/train.record', 'Path to output tfrecord file.')
flags.DEFINE_string('ancher_path', './model/pet_anchors.txt', 'Path to anchors info.')
flags.DEFINE_integer('width', 416, 'Target image width')
flags.DEFINE_integer('height', 416, 'Target image height')
flags.DEFINE_integer('channels', 3, 'Input image channels')
FLAGS = flags.FLAGS

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

def dense_to_sparse_value(a):
    idx = np.nonzero(a)
    values = a[idx]
    idx = np.concatenate([idx], axis=1).transpose()
    sparse = tf.SparseTensorValue(idx, values, a.shape)
    return sparse

def dense_to_sparse_string(dense_shape):
    dense = tf.placeholder(tf.float32, dense_shape)
    idx = tf.where(tf.not_equal(dense, 0))
    sparse = tf.SparseTensor(idx, tf.gather_nd(dense, idx), tf.shape(dense, out_type=tf.int64))
    sparse = tf.io.serialize_sparse(sparse)
    return dense, sparse

def create_tf_example(sess, image_file, xml_file, class_dict, anchors):
    image = cv.imread(image_file)
    raw_h, raw_w = image.shape[:2]
    image = cv.resize(image, (FLAGS.width, FLAGS.height))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    encoded_jpg = cv.imencode('.jpg', image)[1].tostring()
    filename = image_file.split(os.sep)[-1]
    label = 0 if filename[0].isupper() else 1

    y_true = read_boxes(xml_file, class_dict, anchors)

    dense, sparse = dense_to_sparse_string([None, None, anchors.shape[0] // 3, 5 + len(class_dict)])
    y_sparse = []
    for y in y_true:
        y = np.squeeze(y, axis=0)
        sst = sess.run(sparse, feed_dict={dense: y})
        y_sparse.append(sst)
    
    tf_example = tf.train.Example(
        features=tf.train.Features(feature={
            'image': _bytes_feature(encoded_jpg),
            'raw_h': _int64_feature(raw_h),
            'raw_w': _int64_feature(raw_w),
            # 'image/format': _bytes_feature('jpg'.encode()),
            'class': _int64_feature(label),
            'y1': _bytes_list_feature(y_sparse[0]),
            'y2': _bytes_list_feature(y_sparse[1]),
            'y3': _bytes_list_feature(y_sparse[2])}))
    return tf_example


def generate_tfrecord(file_list, image_path, xml_path, class_dict, anchors, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    with tf.Session() as sess:
        for filepath in tqdm(file_list):
            filename = filepath.split('.')[0]
            image_file = os.path.join(FLAGS.image_path, filename + '.jpg')
            xml_file = os.path.join(FLAGS.xml_path, filename + '.xml')
            tf_example = create_tf_example(sess, image_file, xml_file, class_dict, anchors)
            writer.write(tf_example.SerializeToString())
        
    writer.close()
    

def main(_):
    # write 
    classes_name = {'cat': 0, 'dog': 1}
    anchors = read_anchors(FLAGS.ancher_path)
    file_list = os.listdir(FLAGS.xml_path)
    generate_tfrecord(file_list, FLAGS.image_path, FLAGS.xml_path,
        classes_name, anchors, FLAGS.output_path)
    
    
if __name__ == '__main__':
    tf.app.run()
