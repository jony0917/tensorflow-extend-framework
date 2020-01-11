# --coding:utf-8--
import unittest

import tensorflow as tf
import tef_core

class FMModelWarmerTest(unittest.TestCase):
    def test_zero_out(self):

        @tf.function
        def zero_out(x):
            return tef_core.zero_out(x)

        res = zero_out([1,2,3,4,5,6])
        print res

if __name__ == '__main__':
    unittest.main()

