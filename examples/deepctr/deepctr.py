
import multiprocessing
import tensorflow as tf
import tef
import tef.ops
import tef.training


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
                if key == "interest" or key == "ad_kw":
                    while len(values) < 5:
                        values.append(0)
                kv[key] = values

            print kv
            batch_queue.put((kv["uid"][0], kv["age"][0], kv["interest"], kv["aid"][0], kv["ad_kw"], kv["label"]))


def data_generator():
    global batch_queue
    while True:
        yield batch_queue.get()


def data_from_feed():
    data_set = tf.data.Dataset.from_generator(data_generator, (tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.float32))
    #data_set = data_set.padded_batch(4, padded_shapes=[None])
    data_set = data_set.batch(4)
    iterator =  tf.compat.v1.data.make_one_shot_iterator(data_set)
    return iterator.get_next()


def full_connect(name, input, input_dim, output_dim):
    w = tef.ops.variable("%s_w" % name, [input_dim, output_dim], tf.float32)
    b = tef.ops.variable("%s_b" % name, [output_dim], tf.float32)
    return tf.sigmoid(tf.matmul(input, w) + b)


def dense_to_sparse(dense, missing_element):
    indices = tf.where(tf.not_equal(dense, missing_element))
    values = tf.gather_nd(dense, indices)
    shape = tf.shape(dense, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)

def deep_ctr():
    graph = tf.Graph()
    with graph.as_default():
        uid, age, interest, aid, ad_kw, label = data_from_feed()

        embs = []
        uid_emb = tef.ops.embedding(uid, "uid", [20], tf.float32, id_type="hash")
        embs.append(uid_emb)

        age_emb = tef.ops.embedding(age, "age", [120, 20], tf.float32, id_type="index")
        embs.append(age_emb)


        sp_interest = dense_to_sparse(interest, 0)
        interest_emb = tef.ops.embedding_sparse(sp_interest,
                                                "interest",
                                                [20],
                                                tf.float32,
                                                id_type="hash",
                                                combiner="mean")
        embs.append(interest_emb)

        aid_emb = tef.ops.embedding(aid, "aid", [20], tf.float32, id_type="hash")
        embs.append(aid_emb)



        sp_ad_kw = dense_to_sparse(ad_kw, 0)
        ad_kw_emb = tef.ops.embedding_sparse(sp_ad_kw,
                                             "ad_kw",
                                             [20],
                                             tf.float32,
                                             id_type="hash",
                                             combiner="mean")
        embs.append(ad_kw_emb)

        x = tf.concat(embs, axis=1)
        x = full_connect("fc_1", x, 5 * 20, 100)
        x = full_connect("fc_2", x, 100, 100)
        y = full_connect("fc_3", x, 100, 1)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(y, label)
        sgd_optimizer = tef.training.GradientDescentOptimizer(0.002)
        gs, stubs = sgd_optimizer.gradients(loss)
        for i in range(len(gs)):
            print "---------------------------"
            print gs[i]
            print "name=%s, category=%s" % (stubs[i].name, stubs[i].category)

    sess = tf.compat.v1.Session(graph = graph)
    batch = 0
    while batch < 1:
        print "batch=%d" % batch
        print "stubs[0].name=%s" % stubs[1].name
        #r = sess.run(gs)
        #print r
        batch += 1


if __name__ == '__main__':
    data_load_process = multiprocessing.Process(target=load_data)
    data_load_process.daemon = True
    data_load_process.start()

    deep_ctr()
