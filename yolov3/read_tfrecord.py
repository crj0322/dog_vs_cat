import tensorflow as tf
import numpy as np
import cv2 as cv

def parse_fun(serialized_example, shape, class_num, anchor_num):
    # parse single example
    features = tf.parse_single_example(serialized_example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'raw_h': tf.FixedLenFeature([], tf.int64),
        'raw_w': tf.FixedLenFeature([], tf.int64),
        'class': tf.FixedLenFeature([], tf.int64),
        'y1': tf.FixedLenFeature([3], tf.string),
        'y2': tf.FixedLenFeature([3], tf.string),
        'y3': tf.FixedLenFeature([3], tf.string)
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
        y = tf.reshape(y, [shape[0]//scale[i], shape[1]//scale[i], anchor_num, 5+class_num])
        y_true.append(y)

    return [image, *y_true]

def get_dataset(record_path, shape, class_num, anchor_num):
    """Get a tensorflow record file."""
    dataset = tf.data.TFRecordDataset([record_path])
    dataset = dataset.map(lambda x: parse_fun(x, shape, class_num, anchor_num))
    return dataset
    

def main():
    # test read
    dataset = get_dataset('./data/train.record', [416, 416, 3], 2, 3)
    batch = dataset.batch(10).make_one_shot_iterator().get_next()
    img, y1, y2, y3 = batch

    sess = tf.InteractiveSession()
    i = 1
    while True:
        try:
            img, y1, y2, y3 = sess.run(batch)
            y_true = [y1, y2, y3]
            img = (img[0]*255).astype(np.uint8)
        except tf.errors.OutOfRangeError:
            print("End of dataset")
            break
        else:
            cv.imshow('img', img)
            for y in y_true:
                idx = np.nonzero(y)
                values = y[idx]
                print('shape: ', y.shape)
                print('value: ', values)
                print('idx: ', idx)
            cv.waitKey()
        i+=1
    
    
    
if __name__ == '__main__':
    main()
