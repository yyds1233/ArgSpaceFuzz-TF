"""Minimal tf.data callback harness fixture for O7 recipe tests."""
import tensorflow as tf


def _callback(value):
    return tf.math.square(tf.cast(value, tf.int64)) + 1


def TestOneInput(data: bytes):
    ds = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])
    ds = ds.map(
        lambda x: tf.py_function(_callback, [x], tf.int64),
        num_parallel_calls=2,
    )
    for value in ds:
        _ = tf.identity(value)
