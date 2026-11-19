"""Build helper for the low-latency megakernel shared library (``libmk6``).

The kernel uses BF16 ``mma.sync`` tensor-core instructions, ``cp.async`` and a
cooperative grid launch, so it needs compute capability 8.0 or newer.  Each
block also needs ``96,256`` bytes of dynamic plus about 4 KB of static shared
memory (100,368 B in total), i.e. a GPU whose opt-in per-block shared-memory
limit is at least that large.  Consumer Ampere/Ada/Blackwell GPUs (sm_86,
sm_89, sm_120: 101,376 B = 99 KiB) and data-centre parts (sm_80/87: 163 KB,
sm_90/100: 227 KB) qualify; Volta/Turing (sm_70: 96 KB, sm_75: 64 KB) do not.
The paper's timing runs used ``-arch=sm_120`` on an RTX 5080.  On unsupported
GPUs use the stock PyTorch path (``decoder.decode``), which gives the same
decisions at higher latency.

Usage:
    python -m graphqec.decoder.nn.low_latency.build              # detect the GPU, build into the cache
    CUDA_ARCH=sm_120 python -m graphqec.decoder.nn.low_latency.build --out libmk6.so
    python -m graphqec.decoder.nn.low_latency.build --check      # report capability only

Environment: ``CUDA_ARCH`` (e.g. ``sm_89``, ``89``, ``120``) overrides detection,
``NVCC`` or ``CUDA_HOME`` select the compiler, ``NVCC_CCBIN`` selects the host
compiler (``-ccbin``), ``GRAPHQEC_LL_CACHE`` sets the cache directory.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "mk6.cu"
DYNAMIC_SMEM = 96256
STATIC_SMEM_ESTIMATE = 4112  # cudaFuncGetAttributes(mk_kernel).sharedSizeBytes
REQUIRED_SMEM = DYNAMIC_SMEM + STATIC_SMEM_ESTIMATE
MIN_CC = (8, 0)
OPTIN_SMEM_BY_CC = {(7, 0): 98304, (7, 2): 98304, (7, 5): 65536, (8, 0): 166912, (8, 6): 101376, (8, 7): 166912,
                    (8, 9): 101376, (9, 0): 232448, (10, 0): 232448, (12, 0): 101376}


class LowLatencyUnsupported(RuntimeError):
    """The current GPU or toolchain cannot run the low-latency kernel."""


def parse_arch(value: str) -> str:
    v = value.strip().lower().removeprefix("sm_").removeprefix("compute_")
    if not v.isdigit():
        raise ValueError(f"cannot parse CUDA_ARCH={value!r}; expected e.g. sm_89 or 120")
    return f"sm_{v}"


def _cudart():
    names = []
    try:
        import nvidia.cuda_runtime as rt  # pip CUDA runtime shipped with torch
        names += glob.glob(os.path.join(os.path.dirname(rt.__file__), "lib", "libcudart.so*"))
    except ImportError:
        pass
    names += ["libcudart.so.12", "libcudart.so"]
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    return None


def device_report(device: int = 0) -> dict:
    """Compute capability, SM count and shared-memory limits of ``device`` (no compilation)."""
    import torch

    if not torch.cuda.is_available():
        return {"cuda_available": False, "supported": False, "reason": "no CUDA device visible"}
    major, minor = torch.cuda.get_device_capability(device)
    props = torch.cuda.get_device_properties(device)
    optin, coop, source = None, None, "table"
    lib = _cudart()
    if lib is not None:
        v = ctypes.c_int(0)
        if lib.cudaDeviceGetAttribute(ctypes.byref(v), 97, device) == 0:
            optin, source = v.value, "cudaDeviceGetAttribute"
        if lib.cudaDeviceGetAttribute(ctypes.byref(v), 95, device) == 0:
            coop = bool(v.value)
    if optin is None:
        optin = OPTIN_SMEM_BY_CC.get((major, minor))
    reasons = []
    if (major, minor) < MIN_CC:
        reasons.append(f"compute capability {major}.{minor} < 8.0 (needs BF16 mma.sync and cp.async)")
    if optin is not None and optin < REQUIRED_SMEM:
        reasons.append(f"opt-in shared memory per block {optin} B < ~{REQUIRED_SMEM} B required")
    if coop is False:
        reasons.append("device does not support cooperative launches")
    return {"cuda_available": True, "name": props.name, "compute_capability": f"{major}.{minor}",
            "arch": f"sm_{major}{minor}", "sm_count": props.multi_processor_count,
            "smem_optin_per_block": optin, "smem_limit_source": source, "cooperative_launch": coop,
            "kernel_smem_required_estimate": REQUIRED_SMEM, "supported": not reasons,
            "reason": "; ".join(reasons) or None}


def find_nvcc() -> str:
    for cand in (os.environ.get("NVCC"),
                 os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc") if os.environ.get("CUDA_HOME") else None,
                 shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise LowLatencyUnsupported("nvcc not found: install the CUDA toolkit (12.x; 12.8+ for sm_120) or set NVCC/CUDA_HOME")


def nvcc_command(nvcc: str, arch: str, out: Path) -> list[str]:
    cmd = [nvcc]
    if os.environ.get("NVCC_CCBIN"):
        cmd += ["-ccbin", os.environ["NVCC_CCBIN"]]
    return cmd + ["-O3", f"-arch={arch}", "-std=c++17", "-Xcompiler", "-fPIC", "-shared", "-cudart", "static",
                  "-lineinfo", "-o", str(out), str(SRC)]


def cache_dir() -> Path:
    return Path(os.environ.get("GRAPHQEC_LL_CACHE", Path.home() / ".cache" / "graphqec" / "low_latency"))


def build_library(arch: str | None = None, out: Path | None = None, force: bool = False, verbose: bool = True) -> Path:
    """Compile ``mk6.cu`` for ``arch`` (default: ``$CUDA_ARCH`` or the visible GPU) and return the ``.so`` path."""
    if arch is None and os.environ.get("CUDA_ARCH"):
        arch = os.environ["CUDA_ARCH"]
    if arch is None:
        report = device_report()
        if not report.get("cuda_available"):
            raise LowLatencyUnsupported("no CUDA device visible; set CUDA_ARCH to cross-compile")
        if not report["supported"]:
            raise LowLatencyUnsupported(f"{report['name']} ({report['arch']}): {report['reason']}. "
                                        "Use the stock PyTorch decoder (decoder.decode) on this GPU.")
        arch = report["arch"]
    arch = parse_arch(arch)
    if int(arch[3:]) < 80:
        raise LowLatencyUnsupported(f"{arch} is below sm_80; the kernel needs BF16 mma.sync and cp.async")
    nvcc = find_nvcc()
    src_hash = hashlib.sha256(SRC.read_bytes()).hexdigest()[:12]
    if out is None:
        out = cache_dir() / f"libmk6_{arch}_{src_hash}.so"
    out = Path(out)
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = nvcc_command(nvcc, arch, out)
    if verbose:
        print("[low_latency] " + " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        hint = ""
        err = proc.stderr.lower()
        if "unsupported gpu architecture" in err or "not defined for option 'gpu-architecture'" in err:
            hint = f" ({arch} needs a newer CUDA toolkit, e.g. 12.8+ for sm_120)"
        elif "unsupported gnu version" in err:
            hint = " (set NVCC_CCBIN to a supported host compiler, e.g. g++-11)"
        raise LowLatencyUnsupported(f"nvcc failed{hint}:\n{proc.stderr[-4000:]}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default=None, help="override the target, e.g. sm_89 (default: $CUDA_ARCH or the GPU)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true", help="only report device capability")
    args = parser.parse_args(argv)
    report = device_report()
    print(json.dumps(report, indent=1))
    if args.check:
        return 0 if report.get("supported") else 1
    try:
        path = build_library(args.arch, args.out, args.force)
    except LowLatencyUnsupported as exc:
        print(f"UNSUPPORTED: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
