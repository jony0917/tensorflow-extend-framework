# --coding:utf-8--
import unittest

import tensorflow as tf
import tef_core

class FMModelWarmerTest(unittest.TestCase):
    def _test_zero_out(self):
        @tf.function
        def zero_out(x):
            return tef_core.zero_out(x)
        res = zero_out([1,2,3,4,5,6])
        print res

        graph = tf.Graph()
        with graph.as_default():
            input = tf.compat.v1.placeholder(tf.int32)
            output = tef_core.zero_out(input)
        sess = tf.compat.v1.Session(graph = graph)
        res = sess.run(output, feed_dict={input : [2,2,2,2,2]})
        print res


    def test_example_op(self):
        graph = tf.Graph()
        with graph.as_default():
            input = tf.compat.v1.placeholder(tf.int32)
            output = tef_core.example(input)
        sess = tf.compat.v1.Session(graph = graph)
        res = sess.run(output, feed_dict={input : [3,3,3,3,3]})
        print res


if __name__ == '__main__':
    unittest.main()

