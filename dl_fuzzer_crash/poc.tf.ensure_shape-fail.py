# poc.tf.ensure_shape.fatal.py
import ctypes
import numpy as np
import tensorflow as tf

print("TensorFlow:", tf.__version__)

x = np.array([3892, 51006], dtype=np.uint16)
shape = np.empty((0, 18, 3, 2, 10), dtype=np.uint8)

print("x:", repr(x), "dtype=", x.dtype, "shape=", x.shape)
print("shape:", repr(shape), "dtype=", shape.dtype, "shape=", shape.shape)

# 关键：手动设置一个 pending Python exception，
# 但不让 Python 层立刻 raise。
#
# 不要用 ctypes.pythonapi / ctypes.PyDLL，
# 它们通常会检测 PyErr 并立刻转成 Python 异常。
#
# 用 CDLL(None) 直接调用 CPython C API，尽量保留 PyErr 状态，
# 然后马上进入 TensorFlow fast path。
libpython = ctypes.CDLL(None)

PyErr_SetString = libpython.PyErr_SetString
PyErr_SetString.argtypes = [ctypes.py_object, ctypes.c_char_p]
PyErr_SetString.restype = None

PyErr_SetString(RuntimeError, b"poison pending PyErr before TensorFlow fast path")

# 这里不要在中间 print / repr / 做任何 Python 操作
# 直接调用 TensorFlow
tf.ensure_shape(x=x, shape=shape)