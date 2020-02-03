
import tensorflow as tf
import tef
import tef.pywrap
import tef.utils

class Optimizer(object):
    def __init__(self):
        pass



class GradientDescentOptimizer(Optimizer):

    def __init__(self, learning_rate):
        super(GradientDescentOptimizer, self).__init__()
        self.learning_rate = learning_rate


    def gradients(self, loss):
        tef_trainable = tef.utils.get_collection(tef.utils.TEF_TRAINABLE_COLLECTION)
        gs = []
        stubs = []
        for stub in tef_trainable:
            gradient = tf.gradients(loss, stub.var)
            gs.append(gradient)
            stubs.append(stub)
        return gs, stubs


    def apply(self, gs, stubs):
        assert len(gs) == len(stubs)

        push_ops = []
        for i in range(len(gs)):
            gradient = gs[i]
            stub = stubs[i]
            if stub.category == "dense":
                push_op = tef.pywrap.ps_push(gradient,
                                             stub.name,
                                             stub.shape,
                                             stub.dtype,
                                             "SGD",
                                             self.learning_rate)
            elif stub.category == "sparse":
                push_op = tef.pywrap.ps_sparse_push(stub.ids,
                                                    gradient,
                                                    stub.name,
                                                    stub.shape,
                                                    stub.dtype,
                                                    "SGD",
                                                    self.learning_rate)
            elif stub.category == "hash":
                push_op = tef.pywrap.ps_hash_push(stub.ids,
                                                  gradient,
                                                  stub.name,
                                                  stub.shape,
                                                  stub.dtype,
                                                  "SGD",
                                                  self.learning_rate)
            push_ops.apped(push_op)

        return tf.group(push_ops)

    def minimize(self, loss):
        gs, stubs = self.gradients(loss)
        return self.apply(gs, stubs)