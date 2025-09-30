from graphqec.decoder.nn import get_model

__all__ = [
    # "BPOSD",
    # "SlidingWindowBPOSD",
    # "PyMatching",
    # "ConcatMatching",
    "get_model"
]

try:
    from .bposd import BPOSD
    from .slidingwindow_bposd import SlidingWindowBPOSD
    __all__ += ["BPOSD", "SlidingWindowBPOSD"]
except ImportError:
    print("BPOSDDecoder not available. Please install the package 'ldpc' to use this decoder.")
try:
    from .concat_matching import ConcatMatching
    from .pymatching import PyMatching
    __all__ += ["PyMatching", "ConcatMatching"]
except ImportError:
    print("PyMatchingDecoder not available. Please install the package 'pymatching' to use this decoder.")

