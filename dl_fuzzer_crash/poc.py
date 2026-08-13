
import tensorflow as tf


def main():
    print("TensorFlow:", tf.__version__)

    # 关键：inputs 是 scalar，rank=0
    inputs = tf.constant(1.0, dtype=tf.float32)

    # gradient 和 input 必须 same size，所以也设成 scalar
    gradients = tf.constant(1.0, dtype=tf.float32)

    # per-channel min/max，正常应该匹配 inputs 最后一维 channel 数
    # 但 scalar 没有最后一维，C++ 若直接 dims()-1 就会访问 -1
    min_val = tf.constant([0.0], dtype=tf.float32)
    max_val = tf.constant([1.0], dtype=tf.float32)

    print("calling gradient op...")

    y = tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient(
        gradients=gradients,
        inputs=inputs,
        min=min_val,
        max=max_val,
        num_bits=8,
        narrow_range=False,
    )

    print(y)


if __name__ == "__main__":
    main()