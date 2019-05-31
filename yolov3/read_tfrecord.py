import tensorflow as tf
import numpy as np
import cv2 as cv

def parse_fun(serialized_example, shape, class_num, anchor_num):
    # parse single example
    features = tf.io.parse_single_example(serialized_example, features={
        'image': tf.io.FixedLenFeature([], tf.string),
        'raw_h': tf.io.FixedLenFeature([], tf.int64),
        'raw_w': tf.io.FixedLenFeature([], tf.int64),
        'class': tf.io.FixedLenFeature([], tf.int64),
        'y1': tf.io.FixedLenFeature([3], tf.string),
        'y2': tf.io.FixedLenFeature([3], tf.string),
        'y3': tf.io.FixedLenFeature([3], tf.string)
    })

    raw_image = features['image']
    image = tf.image.decode_jpeg(raw_image, channels=shape[-1])
    image = tf.reshape(image, shape)
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255.)

    y_sparse = [features['y1'], features['y2'], features['y3']]
    y_true = []
    scale = [32, 16, 8]
    for i, y in enumerate(y_sparse):
        # tf.io.deserialize_many_sparse() requires the dimensions to be [N,3]
        y = tf.expand_dims(y, axis=0)
        y = tf.io.deserialize_many_sparse(y, dtype=tf.float32)
        y = tf.sparse.to_dense(y)
        # y = tf.squeeze(y, axis=0)
        y = tf.reshape(y, [shape[0]//scale[i], shape[1]//scale[i], anchor_num*(5+class_num)])
        y_true.append(y)

    return image, tuple(y_true)

def get_dataset(record_path, shape, class_num, anchor_num):
    """Get a tensorflow record file."""
    dataset = tf.data.TFRecordDataset([record_path])
    dataset = dataset.map(lambda x: parse_fun(x, shape, class_num, anchor_num))
    return dataset
    

def main():
    # test read
    dataset = get_dataset('./data/train.record', [416, 416, 3], 2, 3)
    batch = dataset.batch(10)
    for img, y_true in batch:
        y_true = [y.numpy() for y in y_true]
        img = (img.numpy()[0]*255).astype(np.uint8)
        cv.imshow('img', img)
        for y in y_true:
            idx = np.nonzero(y)
            values = y[idx]
            print('shape: ', y.shape)
            # print('value: ', values)
            # print('idx: ', idx)
        cv.waitKey()
    
    
if __name__ == '__main__':
    main()
