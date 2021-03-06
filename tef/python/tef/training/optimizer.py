
import tensorflow as tf
import tef
import tef.pywrap
import tef.utils

class BaseOptimizer(object):
    def __init__(self):
        pass

    def compute_gradients(self, loss):
        tef_trainable = tef.utils.get_collection(tef.utils.TEF_TRAINABLE_COLLECTION)
        gs = []
        stubs = []
        for stub in tef_trainable:
            gradient = tf.gradients(loss, stub.var)
            assert len(gradient) == 1
            gs.append(gradient[0])
            stubs.append(stub)
        return gs, stubs

    def apply_gradients(self, gs, stubs):
        """
        To be implemented in subclass.

        :param gs: gradients,  list of tf.Tensor or tf.IndexedSlices object.
        :param stubs: tef variable stubs
        :return: train operation
        """

        assert False

    def minimize(self, loss):
        gs, stubs = self.compute_gradients(loss)
        return self.apply_gradients(gs, stubs)


class GradientDescentOptimizer(BaseOptimizer):

    def __init__(self, learning_rate):
        super(GradientDescentOptimizer, self).__init__()
        self.learning_rate = learning_rate

    def apply_gradients(self, gs, stubs):
        assert len(gs) == len(stubs)

        push_ops = []
        for i in range(len(gs)):
            gradient = gs[i]
            stub = stubs[i]
            if stub.category == "dense":
                assert isinstance(gradient, tf.Tensor)
                push_op = tef.pywrap.ps_push(gradient,
                                             stub.name,
                                             stub.shape,
                                             stub.dtype,
                                             "SGD",
                                             self.learning_rate)
            elif stub.category == "index":
                assert isinstance(gradient, tf.IndexedSlices)
                ids = tf.gather(stub.ids, gradient.indices)
                push_op = tef.pywrap.ps_sparse_push(ids,
                                                    gradient.values,
                                                    stub.name,
                                                    stub.shape,
                                                    stub.dtype,
                                                    "SGD",
                                                    self.learning_rate)
            elif stub.category == "hash":
                assert isinstance(gradient, tf.IndexedSlices)
                ids = tf.gather(stub.ids, gradient.indices)
                push_op = tef.pywrap.ps_hash_push(ids,
                                                  gradient.values,
                                                  stub.name,
                                                  stub.shape,
                                                  stub.dtype,
                                                  "SGD",
                                                  self.learning_rate)
            else:
                assert False

            push_ops.append(push_op)

        return tf.group(push_ops)

