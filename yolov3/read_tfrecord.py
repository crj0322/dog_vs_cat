import tensorflow as tf
import numpy as np
import cv2 as cv

def parse_fun(serialized_example, shape, class_num, anchor_num, layer_num):
    # parse single example
    feature_dict = {'image': tf.io.FixedLenFeature([], tf.string),
        'raw_h': tf.io.FixedLenFeature([], tf.int64),
        'raw_w': tf.io.FixedLenFeature([], tf.int64),
        'class': tf.io.FixedLenFeature([], tf.int64)}

    for i in range(layer_num):
        feature_dict['y%d' % i] = tf.io.FixedLenFeature([3], tf.string)
    
    features = tf.io.parse_single_example(serialized_example, features=feature_dict)

    raw_image = features['image']
    image = tf.image.decode_jpeg(raw_image, channels=shape[-1])
    image = tf.reshape(image, shape)
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255.)

    y_sparse = [features['y%d' % i] for i in range(layer_num)]
    y_true = []
    
    # e.g. [32, 16, 8]
    scale = [2**(5 - i) for i in range(layer_num)]
    for i, y in enumerate(y_sparse):
        # tf.io.deserialize_many_sparse() requires the dimensions to be [N,3]
        y = tf.expand_dims(y, axis=0)
        y = tf.io.deserialize_many_sparse(y, dtype=tf.float32)
        y = tf.sparse.to_dense(y)
        # y = tf.squeeze(y, axis=0)
        y = tf.reshape(y, [shape[0]//scale[i], shape[1]//scale[i], anchor_num*(5+class_num)])
        y_true.append(y)

    return image, tuple(y_true)

def get_dataset(record_path, shape, class_num, anchor_num, layer_num):
    """Get a tensorflow record file."""
    dataset = tf.data.TFRecordDataset([record_path])
    dataset = dataset.map(lambda x: parse_fun(x, shape, class_num, anchor_num, layer_num))
    return dataset
    

def main():
    # test read
    dataset = get_dataset('./data/train_tiny.record', [416, 416, 3], 
        class_num=2, anchor_num=3, layer_num=2)
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
