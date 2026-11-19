from graphqec.qecc.code import *
from graphqec.qecc.color_code.sydney_color_code import TriangleColorCode
from graphqec.qecc.ldpc_code.bbcode import ETHBBCode
from graphqec.qecc.ldpc_code.shyps import SHYPSCode
from graphqec.qecc.ldpc_code.toric4d import Toric4DCode
from graphqec.qecc.surface_code.stim_block_memory import RotatedSurfaceCode
# $QEC_DATA_PATH is only read when a SycamoreSurfaceCode object is built.
from graphqec.qecc.surface_code.google_block_memory import SycamoreSurfaceCode, QECDataGoogleFormat
from graphqec.qecc.surface_code._data_path import MissingExperimentDataError

__all__ = [
    'QuantumCode',
    'TannerGraph',
    'TemporalTannerGraph',
    'SycamoreSurfaceCode',
    'TriangleColorCode',
    'ETHBBCode',
    'RotatedSurfaceCode',
    'SHYPSCode',
    "Toric4DCode",
    'MissingExperimentDataError',
    'get_code'
]

def get_code(name, **kwargs):
    target = globals()[name]
    if issubclass(target, QuantumCode):
        if "profile_name" in kwargs:
            return target.from_profile(**kwargs)
        else:
            return target(**kwargs)
    else:
        raise ValueError("Invalid code name")

