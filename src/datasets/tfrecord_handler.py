from typing import List

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class TFRecordHandler:
    @staticmethod
    def _bytes_feature(value):
        """Return a bytes_list from a string / byte.
        From:
        https://www.tensorflow.org/tutorials/load_data/tfrecord
        """
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Return a float_list from a float / double.
        From:
        https://www.tensorflow.org/tutorials/load_data/tfrecord
        """
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Return an int64_list from a bool / enum / int / uint.
        From:
        https://www.tensorflow.org/tutorials/load_data/tfrecord
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def serialize_example(image: np.array, fine_label: int, coarse_label: int) -> str:
        """Create a tf.train.Example message ready to be written to a file.
        From:
        https://www.tensorflow.org/tutorials/load_data/tfrecord
        """
        image_shape = tf.shape(image)

        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        feature = {
            # 'height': TFRecordHandler._int64_feature(image_shape[0]),
            # 'width': TFRecordHandler._int64_feature(image_shape[1]),
            # 'depth': TFRecordHandler._int64_feature(image_shape[2]),
            'image_raw': TFRecordHandler._bytes_feature(tf.io.serialize_tensor(image)),  # Serialize array to string
            'fine_label': TFRecordHandler._int64_feature(fine_label),
            'coarse_label': TFRecordHandler._int64_feature(coarse_label),
        }

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    @staticmethod
    def write_examples(file_name: str, image: List[np.array], fine_label: List[int], coarse_label: List[int]):
        with tf.io.TFRecordWriter(file_name) as writer:
            for img, fine, coarse in zip(image, fine_label, coarse_label):
                serialized = TFRecordHandler.serialize_example(img, fine, coarse)
                writer.write(serialized)

    @staticmethod
    def _parse_tfr_element(element):
        """Parse a single example from a TFRecord.
        Based on:
        https://stackoverflow.com/a/60283571 and https://www.tensorflow.org/tutorials/load_data/tfrecord
        """
        feature_description = {
            # 'height': tf.io.FixedLenFeature([], tf.int64),
            # 'width': tf.io.FixedLenFeature([], tf.int64),
            # 'depth': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'fine_label': tf.io.FixedLenFeature([], tf.int64),
            'coarse_label': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(element, feature_description)

        # Process the image to an array
        image_raw = example.pop('image_raw')  # Get byte string
        example['image'] = tf.io.parse_tensor(image_raw, out_type=tf.uint8)  # Restore the array from byte string

        # Restore the shape of the image
        example['image'] = tf.ensure_shape(example['image'], (None, None, None))

        return example

    @staticmethod
    def read_examples(file_name: str) -> tf.data.TFRecordDataset:
        """
        Based on:
        https://www.tensorflow.org/tutorials/load_data/tfrecord
        """
        raw_dataset = tf.data.TFRecordDataset(file_name)
        parsed_dataset = raw_dataset.map(TFRecordHandler._parse_tfr_element, num_parallel_calls=AUTOTUNE)
        return parsed_dataset

    @staticmethod
    def count_size(dataset: tf.data.TFRecordDataset) -> int:
        count = 0
        for _ in dataset:
            count += 1
        return count
