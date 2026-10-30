"""Unknown-contract fixture used by the explore-mode regression test."""
import atheris
import tensorflow as tf


@atheris.instrument_func
def TestOneInput(data: bytes):
    fdp = atheris.FuzzedDataProvider(data)
    n = fdp.ConsumeIntInRange(1, 4)
    x = tf.constant([fdp.ConsumeFloat() for _ in range(n)], dtype=tf.float32)
    y = tf.math.exp(x)
    _ = tf.identity(y)
