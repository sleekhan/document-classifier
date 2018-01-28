from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

# Seed for repeatability.
_RANDOM_SEED = 0


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    doc_root = os.path.join(dataset_dir, "docs_images")
    directories = []
    class_names = []
    for filename in os.listdir(doc_root):
        path = os.path.join(doc_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    doc_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            doc_filenames.append(path)

    return doc_filenames, sorted(class_names)


def _get_dataset_filename(dataset_name, dataset_dir, split_name, shard_id, num_shards):
    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
        dataset_name, split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_name, dataset_dir, num_shards):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_name: This is a prefix name of TFRecords to be converted
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(num_shards):
                output_filename = _get_dataset_filename(
                    dataset_name, dataset_dir, split_name, shard_id, num_shards)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def _dataset_exists(dataset_dir, num_shards):
    for split_name in ['train', 'validation']:
        for shard_id in range(num_shards):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def run(dataset_name, dataset_dir, num_shards, ratio_val):
    """Runs the download and conversion operation.

    """
    if not tf.gfile.Exists(dataset_dir):
        print('The dataset directory does not exist. Please specify the data directory again.')
        return

    if _dataset_exists(dataset_dir, num_shards):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    doc_filenames, class_names = _get_filenames_and_classes(dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # Cacluate number of validation proportional to ratio_val
    num_validation = int(len(doc_filenames) * ratio_val)

    # Divide into train and test:
    random.seed(_RANDOM_SEED)
    random.shuffle(doc_filenames)
    training_filenames = doc_filenames[num_validation:]
    validation_filenames = doc_filenames[:num_validation]

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids, dataset_name, dataset_dir, num_shards)
    _convert_dataset('validation', validation_filenames, class_names_to_ids, dataset_name, dataset_dir, num_shards)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    print('\nFinished converting the Flowers dataset!')
