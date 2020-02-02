
import tef

def variable(name, shape, dtype):
    v = tef.pywrap.ps_pull(name, shape, dtype)
    return v
