
import multiprocessing
import tensorflow as tf
import tef
import tef.ops


batch_queue = multiprocessing.Queue(maxsize=5000)

def load_data():
    global batch_queue
    with open("data.txt") as fp:
        for line in fp.readlines():
            columns = line.split(",")
            assert len(columns) == 6

            kv = {}
            for i in range(len(columns)):
                column = columns[i].strip()
                items = column.split(":")
                assert len(items) == 2
                key = items[0]
                values = items[1].split("|")
                assert len(values) > 0
                for k in range(len(values)):
                    values[k] = int(values[k])
                kv[key] = values

            print kv
            batch_queue.put((kv["uid"], kv["age"], kv["interest"], kv["aid"], kv["ad_kw"], kv["label"]))


def data_generator():
    global batch_queue
    while True:
        yield batch_queue.get()


def data_from_feed():
    data_set = tf.data.Dataset.from_generator(data_generator, (tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.float32))
    #data_set = data_set.padded_batch(4, padded_shapes=[None])
    iterator =  tf.compat.v1.data.make_one_shot_iterator(data_set)
    return iterator.get_next()


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

    uid, age, interest, aid, ad_kw, label = data_from_feed()

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

    sp_ad_kw = dense_to_sparse(ad_kw, 0)
    sp_ad_kw_weight = tf.SparseTensor(sp_ad_kw.indices,
                                            tf.one(sp_ad_kw.values.shape),
                                            sp_ad_kw.shape),
    ad_kw_emb = tef.ops.embedding_sparse(sp_ad_kw,
                                               sp_ad_kw_weight,
                                               "ad_kw",
                                               [20],
                                               tf.float32,
                                               id_type="hash",
                                               combiner="mean")
    embs.append(ad_kw_emb)
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
    data_load_process = multiprocessing.Process(target=load_data)
    data_load_process.daemon = True
    data_load_process.start()

    deep_ctr()
