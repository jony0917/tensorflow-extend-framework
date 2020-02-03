
import tensorflow as tf
import tef.pywrap
import tef.utils



def embedding(ids, name, shape, dtype, id_type="index"):
    ids, idx = tf.unique(ids)
    if id_type == "index":
        emb = tef.pywrap.ps_sparse_pull(ids, name, shape, dtype)
        tef.utils.add_to_collection(tef.utils.TEF_TRAINABLE_COLLECTION,
                                    tef.utils.VariableSub(emb, name, shape, dtype, ids, "index"))
    elif id_type == "hash":
        emb = tef.pywrap.ps_hash_pull(ids, name, shape, dtype)
        tef.utils.add_to_collection(tef.utils.TEF_TRAINABLE_COLLECTION,
                                    tef.utils.VariableSub(emb, name, shape, dtype, ids, "hash"))
    else:
        assert False

    emb = tf.gather(emb, idx)
    return emb


def embedding_sparse(sp_ids, sp_weights, name, shape, dtype, id_type="index", combiner="mean"):
    ids, idx = tf.unique(sp_ids.values)
    if id_type == "index":
        emb = tef.pywrap.ps_sparse_pull(ids, name, shape, dtype)
        tef.utils.add_to_collection(tef.utils.TEF_TRAINABLE_COLLECTION,
                                    tef.utils.VariableSub(emb, name, shape, dtype, ids, "index"))
    elif id_type == "hash":
        emb = tef.pywrap.ps_hash_pull(ids, name, shape, dtype)
        tef.utils.add_to_collection(tef.utils.TEF_TRAINABLE_COLLECTION,
                                    tef.utils.VariableSub(emb, name, shape, dtype, ids, "hash"))

    emb = tf.gather(emb, idx)
    emb *= sp_weights.values

    segment_ids = sp_ids.indices[:, 0]
    if combiner == "sum":
        emb = tf.math.segment_sum(emb, segment_ids)
    elif combiner == "mean":
        emb = tf.math.segment_sum(emb, segment_ids)
        weight_sum = tf.math.segment_sum(sp_weights.values, segment_ids)
        emb = tf.math.divide(emb, weight_sum)
    else:
        assert False

    return emb
