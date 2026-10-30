"""Small TensorFlow harness fixture used by the portable unit tests.

It deliberately keeps a clear target call and deterministic input construction
so loader/analyzer/transformer tests do not depend on /root experiment data.
"""
import atheris
import tensorflow as tf


@atheris.instrument_func
def TestOneInput(data: bytes):
    fdp = atheris.FuzzedDataProvider(data)
    dtype = fdp.PickValueInList([tf.float32, tf.float64])
    n = fdp.ConsumeIntInRange(1, 4)
    values = [fdp.ConsumeFloat() for _ in range(n)]
    x = tf.constant(values, dtype=dtype)
    y = tf.math.log1p(x)
    _ = tf.identity(y)
