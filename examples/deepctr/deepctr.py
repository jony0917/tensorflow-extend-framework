
import tensorflow as tf
import tef

def data_from_feed():
    # uid:987, age:20 interest:2|12|234, aid:1912230, ad_category:car, label:1
    #
    #
    return None, None, None

def full_connect(input, input_dim, output_dim):
    w = tef.variable([input_dim, output_dim], tf.float)
    b = tef.variable([output_dim], tf.float)
    return tf.sigmoid(tf.matmul(input, w) + b)

def dense_to_sparse(dense, missing_element):
    indices = tf.where(tf.not_equal(dense, missing_element))
    values = tf.gather_nd(dense, indices)
    shape = dense.get_shape()
    return tf.SparseTensor(indices, values, shape)

def deep_ctr():
    uid, age, interest, aid, ad_category, label = data_from_feed()

    embs = []
    uid_emb = tef.ops.embedding(uid, "uid", [20], tf.float32, id_type="hash")
    embs.append(uid_emb)

    age_emb = tef.ops.embedding(age, "age", [120, 20], tf.float32, id_type="index")
    embs.append(age_emb)

    sp_interest = dense_to_sparse(interest, 0)
    sp_interest_weight = tf.SparseTensor(sp_interest.indices,
                                         tf.ones(sp_interest.values.shape),
                                         sp_interest.shape)
    interest_emb = tef.ops.embedding_sparse(sp_interest,
                                            sp_interest_weight,
                                            "interest",
                                            [20],
                                            tf.float32,
                                            id_type="hash",
                                            combiner="mean")
    embs.append(interest_emb)

    aid_emb = tef.ops.embedding(aid, "aid", [20], tf.float32, id_type="hash")
    embs.append(aid_emb)

    sp_ad_category = dense_to_sparse(ad_category, 0)
    sp_ad_category_weight = tf.SparseTensor(sp_ad_category.indices,
                                            tf.one(sp_ad_category.values.shape),
                                            sp_ad_category.shape),
    ad_category_emb = tef.ops.embedding_sparse(sp_ad_category,
                                               sp_ad_category_weight,
                                               "ad_category",
                                               [20],
                                               tf.float32,
                                               id_type="hash",
                                               combiner="mean")
    embs.append(ad_category_emb)
    x = tf.concat(embs)
    x = full_connect(x, 5 * 20, 256)
    x = full_connect(x, 256, 128)
    y = full_connect(x, 128, 1)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(y, label)
    sgd_optimizer = tef.training.GradientDescentOptimizer(0.002)
    train_op = sgd_optimizer.minimize(loss)
    sess = tf.compat.v1.session()
    while True:
        sess.run(train_op)



if __name__ == '__main__':
    deep_ctr()
