# --coding:utf-8--
import unittest
import tensorflow as tf


class FMModelWarmerTest(unittest.TestCase):
    def test_zero_out_op(self):
        tef_core = tf.load_op_library('/github/tensorflow-extend-framework/tef/python/tef/pywrap/libtef_core.so')
        pass



if __name__ == '__main__':
    unittest.main()

