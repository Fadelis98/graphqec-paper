from graphqec.decoder.nn.models import *

__all__ = [
    "QECCDecoder",
    "GraphRNNDecoderV5A",
    "HardwareEfficientGraphRNNDecoderV5A",
    "get_model"
    ]

if FLA_ENABLED:
    __all__.append("GraphLinearAttnDecoderV2A")

def get_model(name: str, **kwargs):
    target = globals()[name]
    if issubclass(target, QECCDecoder):
        return target(**kwargs)
    else:
        raise ValueError(f"Invalid model name {name}")
