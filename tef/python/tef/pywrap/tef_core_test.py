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
        config = tf.compat.v1.ConfigProto(allow_soft_placement=False,
                                          log_device_placement=True)
        sess = tf.compat.v1.Session(graph = graph, config=config)
        res = sess.run(output, feed_dict={input : [3,3,3,3,3]})
        print res


    def test_example_op_on_gpu(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/GPU"):
                input = tf.compat.v1.placeholder(tf.int32)
                output = tef_core.example(input)
        config = tf.compat.v1.ConfigProto(allow_soft_placement=False,
                                          log_device_placement=True)
        sess = tf.compat.v1.Session(graph = graph, config = config)
        res = sess.run(output, feed_dict={input : [4,4,4,4,4]})
        print res

if __name__ == '__main__':
    unittest.main()

