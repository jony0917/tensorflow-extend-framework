
import tensorflow as tf
import tef

def data_from_feed():
    pass

def full_connect(input, input_dim, output_dim):
    w = tef.variable([input_dim, output_dim], tf.float)
    b = tef.variable([output_dim], tf.float)
    return tf.sigmoid(tf.matmul(input, w) + b)

def deep_ctr():
    x1, x2, x3_indices, x3_values, x3_shape, w, label = data_from_feed()
    embs = []
    emb1 = tef.embedding.embedding("x1", x1)
    embs.append(emb1)
    emb2 = tef.embedding.embedding("x2", x2)
    embs.append(emb2)

    sp_ids = tf.SparseTensor(x3_indices, x3_values, x3_shape)
    emb3 = tef.embedding.embedding_sparse("x3", sp_ids)
    embs.append(emb3)

    x = tf.concat(embs)
    x = full_connect(x, 3 * 20, 256)
    x = full_connect(x, 256, 128)
    y = full_connect(x, 128, 1)
    losses = tf.nn.weighted_cross_entropy_with_logits(label, x, w)
    loss_op = tf.reduce_mean(losses)
    sgd_optimizer = tef.train.GradientDescentOptimizer(0.002)
    train_op = sgd_optimizer.minimize(losses)
    sess = tf.compat.v1.session()
    while True:
        sess.run([train_op, loss_op])

if __name__ == '__main__':
    deep_ctr()
