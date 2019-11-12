import tensorflow as tf


def get_int_64_feature( value ):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_bytes_feature( value ):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(x, y, filepath):
    writer = tf.python_io.TFRecordWriter(filepath)

    for i in range(x.shape):
        feature = {
            "image": get_bytes_feature(x[i].tostring()),
            "label": get_int_64_feature(y[i])
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example)
