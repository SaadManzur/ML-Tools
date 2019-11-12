import tensorflow as tf


def get_int_64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def parse_function(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }

    return tf.io.parse_single_example(example, feature_description)


def convert_to_tfrecord(x, y, filepath):
    writer = tf.io.TFRecordWriter(filepath)

    for i in range(x.shape[0]):
        feature = {
            "image": get_bytes_feature(x[i].tostring()),
            "label": get_int_64_feature(y[i])
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())


def verify_tfrecord(filepath):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset.map(parse_function)

    for example in dataset.take(10):
        print(repr(example))