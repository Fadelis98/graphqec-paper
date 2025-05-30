from graphqec.qecc.code import *
from graphqec.qecc.color_code.sydney_color_code import *
from graphqec.qecc.ldpc_code.bbcode import *

__all__ = [
    'QuantumCode',
    'TannerGraph',
    'TemporalTannerGraph',
    'ZuchongzhiSurfaceCode',
    'TriangleColorCode',
    'ETHBBCode',
    'get_code'
]

try:
    from .surface_code.google_block_memory import *
    __all__.append('SycamoreSurfaceCode')
except ImportError as e:
    print(e)
    print('qecc.surface_code.google_block_memory not imported')

def get_code(name, **kwargs):
    target = globals()[name]
    if issubclass(target, QuantumCode):
        if "profile_name" in kwargs:
            return target.from_profile(**kwargs)
        else:
            return target(**kwargs)
    else:
        raise ValueError("Invalid code name")

