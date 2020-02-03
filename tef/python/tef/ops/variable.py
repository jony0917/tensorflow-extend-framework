
import tef
import tef.pywrap
import tef.utils

def variable(name, shape, dtype):
    v = tef.pywrap.ps_pull(name, shape, dtype)
    tef.utils.add_to_collection(tef.utils.TEF_TRAINABLE_COLLECTION,
                                tef.utils.VariableSub(v, name, shape, dtype, ids, "dense"))
    return v
