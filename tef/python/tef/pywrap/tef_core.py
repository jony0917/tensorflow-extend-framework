import tensorflow as tf
import os

tef_core = tf.load_op_library(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libtef_core.so'))
zero_out = tef_core.zero_out
example = tef_core.example




