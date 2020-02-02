
import tensorflow as tf
import tef.pywrap



def embedding(ids, name, shape, dtype, id_type="index"):
    ids, idx = tf.unique(ids)
    if id_type == "index":
        emb = tef.pywrap.ps_sparse_pull(ids, name, shape, dtype)
    elif id_type == "hash":
        emb = tef.pywrap.ps_hash_pull(ids, name, shape, dtype)
    else:
        assert False

    emb = tf.gather(emb, idx)
    return emb


def embedding_sparse(sp_ids, sp_weights, name, shape, dtype, id_type="index", combiner="mean"):
    ids, idx = tf.unique(sp_ids.values)
    if id_type == "index":
        emb = tef.pywrap.ps_sparse_pull(ids, name, shape, dtype)
    elif id_type == "hash":
        emb = tef.pywrap.ps_hash_pull(ids, name, shape, dtype)

    emb = tf.gather(emb, idx)
    emb *= sp_weights.values

    segment_ids = sp_ids.indices[:, 0]
    if combiner == "sum":
        emb = tf.segment_sum(emb, segment_ids)
    elif combiner == "mean":
        emb = tf.segment_sum(emb, segment_ids)
        weight_sum = tf.segment_sum(sp_weights.values, segment_ids)
        emb = tf.math.div(emb, weight_sum)
    else:
        assert False

    return emb
