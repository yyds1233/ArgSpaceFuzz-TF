"""Minimal tf.data Dataset.map harness fixture for O6 recipe tests."""
import tensorflow as tf


def TestOneInput(data: bytes):
    ds = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])
    ds = ds.map(lambda x: tf.math.square(x) + 1, num_parallel_calls=2)
    for value in ds:
        _ = tf.identity(value)
