

GLOBAL_COLLECTIONS = {}
TEF_TRAINABLE_COLLECTION = "tef_trainable_collection"


class VariableSub(object):
    def __init__(self, var, var_name, var_shape, dtype, ids=None, category="dense"):
        self.var = var
        self.var_name = var_name
        self.var_shape = var_shape
        self.dtype = dtype
        self.ids = ids
        self.category = category


def add_to_collection(name, stub):
    global GLOBAL_COLLECTIONS
    if not GLOBAL_COLLECTIONS.has_key(name):
        GLOBAL_COLLECTIONS[name] = []
    GLOBAL_COLLECTIONS.append(stub)



def get_collection(name):
    global GLOBAL_COLLECTIONS
    if GLOBAL_COLLECTIONS.has_key(name):
        return GLOBAL_COLLECTIONS[name]
    else:
        return None
