from graphqec.decoder.nn import get_model

__all__ = ["get_model"]

try:
    from .bposd import BPOSD
    __all__ += ["BPOSD"]
except ImportError as exc:
    # Keep this non-fatal (training and the GraphQEC-only paths do not need it),
    # but report the underlying reason: the bare ImportError would otherwise
    # surface much later as a confusing "cannot import name 'BPOSD'".
    print(f"BPOSDDecoder not available ({exc}). "
          "Install the 'ldpc' package (declared dependency) to enable it.")
try:
    from .concat_matching import ConcatMatching
    from .pymatching import PyMatching
    __all__ += ["PyMatching", "ConcatMatching"]
except ImportError as exc:
    print(f"PyMatchingDecoder not available ({exc}). "
          "Install the 'pymatching' package (declared dependency) to enable it.")

