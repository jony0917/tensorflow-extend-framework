import tensorflow as tf
import os

tef_core_lib = tf.load_op_library(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libtef_core.so'))
zero_out = tef_core_lib.zero_out
example = tef_core_lib.example
ps_pull = tef_core_lib.ps_pull
ps_push = tef_core_lib.ps_push
ps_sparse_pull = tef_core_lib.ps_sparse_pull
ps_sparse_push = tef_core_lib.ps_sparse_push
ps_hash_pull = tef_core_lib.ps_hash_pull
ps_hash_push = tef_core_lib.ps_hash_push
