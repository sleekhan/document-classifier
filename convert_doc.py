# This module will convert jpeg document files to TFRecords for tensorflow

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import convert_image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    'docs',
    'The name of the dataset prefix.')

tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords are saved.')

tf.app.flags.DEFINE_integer(
    'num_shards',
    5,
    'A number of sharding for TFRecord files(integer).')

tf.app.flags.DEFINE_float(
    'ratio_val',
    0.2,
    'A ratio of validation datasets for TFRecord files(flaot, 0 ~ 1).')


def main(_):
    if not FLAGS.dataset_name:
        raise ValueError('You must supply the dataset name with --dataset_name')

    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    convert_image.run(FLAGS.dataset_name, FLAGS.dataset_dir, FLAGS.num_shards, FLAGS.ratio_val)


if __name__ == '__main__':
    tf.app.run()
