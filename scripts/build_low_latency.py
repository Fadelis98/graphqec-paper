"""Build (or check) the low-latency kernel library for the visible GPU; see graphqec/decoder/nn/low_latency/build.py.

    python scripts/build_low_latency.py --check
    CUDA_ARCH=sm_120 python scripts/build_low_latency.py --out libmk6.so
"""
import sys

from graphqec.decoder.nn.low_latency.build import main

if __name__ == "__main__":
    sys.exit(main())
