
import tensorflow as tf
import tef

class Optimizer(object):

    def __init__(self):
        pass



class GradientDescentOptimizer(Optimizer):

    def __init__(self, learning_rate):
        super(GradientDescentOptimizer, self).__init__()
        self.learning_rate = learning_rate

    def minimize(self, loss):
        var_collection = tef.utils.get_collection(TEF_VARIABLE)

        push_ops = []
        for var in var_collection:
            gradient = tf.gradients(loss, var.value)
            if var.type == "dense":
                push_op = tef.pywrap.ps_push(gradient, var.name, var.shape, var.dtype, "SGD", self.learning_rate)

            elif var.type == "sparse":
                push_op = tef.pywrap.ps_sparse_push(var.ids, gradient, var.name, var.shape, var.dtype, "SGD", self.learning_rate)

            elif var.type == "hash":
                push_op = tef.pywrap.ps_hash_push(var.ids, gradient, var.name, var.shape, var.dtype, "SGD", self.learning_rate)

            push_ops.apped(push_op)

        return tf.group(push_ops)

