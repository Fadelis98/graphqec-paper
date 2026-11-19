"""Layout-free detector-role labelling for the GARI-NMS wrapper.

Why this exists
---------------
GARI's upstream ``det_and_err_encoding`` decides which detectors belong to the
X sector and which to the Z sector, and everything downstream (``hx``/``hz``,
``i_hx_only``/``i_hz_only``/``i_hy_only``, ``big_matrix``) follows from that
choice. It picks the roles with helpers that only fit two layouts:

* even distance  -> ``syndrome_index_bb``, a positional tiling driven by
  ``circuit.num_qubits // 4`` that never looks at the DEM;
* odd distance   -> ``syndrome_index_sc``, a rotated-surface-code coordinate
  grid, and that branch is **dead code**: it calls ``syndrome_mask(d, d)``
  although ``syndrome_mask`` takes a single argument, so it raises TypeError.

In the first non-BB baseline batch every job was submitted with ``distance = 12``
(the wrapper's fallback), so every code family was forced through the BB tiling.
Measured consequences (summarised in Supplementary Section 4 of the paper):

* BB72 and the five triangle colour codes happen to get a correct labelling
  (the colour codes differ only by an overall 1<->3 relabel, which is harmless);
* Toric4D18, Toric4D54 and SHYPS49 get an arbitrary permutation of the roles.

What this module does
---------------------
``stim.Circuit.detecting_regions()`` answers the same question directly from
error propagation: for each detector it lists the Pauli errors that flip it, so
a detector whose regions are X-only is one that responds to X errors. That needs
one stim pass and makes no assumption about the circuit layout.

Validation: on BB72 -- the layout the upstream helpers were written for -- this
labelling agrees with the layout path on 396/396 non-ambiguous detectors. Using
it on Toric4D18 drops the logical error rate at p=0.003 from 51.05% (layout
path) to 5.30%, better than Relay-BP's 7.10%, and runs ~400x faster.

``encode_with`` runs the rest of the upstream encoding with a caller-supplied
labelling; that part is role-agnostic. It also skips the two ``nullspace`` calls
the upstream function performs only to return ``None`` for both.
"""

import os

import numpy as np
from ldpc.mod2 import nullspace

# Upstream index convention, fixed empirically rather than by the (misleading)
# comments in helper_functions.py:
#   det_index == 1  ->  the detector's detecting regions are X-only
#   det_index == 3  ->  the detector's detecting regions are Z-only
INDEX_ONE = 1
INDEX_THREE = 3


def det_index_from_regions(circuit):
    """Return ``(index, n_ambiguous)``.

    ``index`` has one entry per detector; ``0`` marks a detector whose regions
    mix X and Z letters, for which no basis can be inferred this way.
    """
    regions = circuit.detecting_regions()
    n_det = circuit.num_detectors
    index = np.zeros(n_det, dtype=int)
    for target, per_tick in regions.items():
        # ``detecting_regions`` also returns one entry per logical observable
        # (DemTarget 'L*'); keep only the detector targets ('D*').
        if not getattr(target, "is_relative_detector_id", lambda: False)():
            continue
        det = int(getattr(target, "val", target))
        if not 0 <= det < n_det:
            continue
        letters = set()
        for value in per_tick.values():
            for pauli in (value if isinstance(value, (list, tuple)) else [value]):
                # str(PauliString) is sign-prefixed, e.g. "+_ZZ_Z__ZZ____".
                letters |= set(str(pauli)[1:]) - {"_"}
        has_x, has_z = bool(letters & {"X"}), bool(letters & {"Z"})
        if has_x and not has_z:
            index[det] = INDEX_ONE
        elif has_z and not has_x:
            index[det] = INDEX_THREE
    return index, int((index == 0).sum())


def encode_with(matrices, circuit, det_index):
    """Upstream ``det_and_err_encoding`` body with a caller-supplied labelling.

    Mirrors helper_functions.py, minus the two ``nullspace`` calls that the
    upstream function computes and then discards.
    """
    h = matrices.check_matrix.toarray()
    det_encoding = np.where(det_index == INDEX_THREE, 1, 0) + np.where(
        det_index == INDEX_ONE, 3, 0)
    hx = h[np.where(det_index == INDEX_ONE)[0]]
    hz = h[np.where(det_index == INDEX_THREE)[0]]
    i_hx = np.unique(np.nonzero(hx)[1])
    i_hz = np.unique(np.nonzero(hz)[1])
    i_hy = np.intersect1d(i_hx, i_hz)
    i_hx_only = np.setdiff1d(i_hx, i_hy)
    i_hz_only = np.setdiff1d(i_hz, i_hy)
    i_hy_only = i_hy
    err_encoding = np.zeros(h.shape[1], dtype="int")
    err_encoding[i_hx_only] = 3
    err_encoding[i_hy_only] = 2
    err_encoding[i_hz_only] = 1

    dx = hx[:, i_hx_only]
    dz = hz[:, i_hz_only]
    from helper_functions import a_in_b_idx, data_encoding
    i_dx_in_hx = a_in_b_idx(dx, hx)
    i_dz_in_hz = a_in_b_idx(dz, hz)
    hx_yonly = hx[:, i_hy_only]
    hz_yonly = hz[:, i_hy_only]
    i_dx_in_hx_yonly = a_in_b_idx(dx, hx_yonly)
    i_dz_in_hz_yonly = a_in_b_idx(dz, hz_yonly)

    m, n = h.shape
    mx, nx = dx.shape
    _mz, nz = dz.shape
    big_matrix = np.zeros((m + nx + nz, n + nx + nz), "int")
    big_matrix[:mx, :nx] = dx
    big_matrix[mx:m, nx:nx + nz] = dz
    big_matrix[m:, :nx + nz] = np.eye(nx + nz, dtype="int")
    big_matrix[m:, nx + nz:2 * (nx + nz)] = np.eye(nx + nz, dtype="int")
    y_part = big_matrix[m:, 2 * (nx + nz):]
    for i, c in enumerate(i_dx_in_hx_yonly):
        y_part[c, i] = 1
    for i, c in enumerate(i_dz_in_hz_yonly):
        y_part[c + nx, i] = 1
    big_matrix[m:, 2 * (nx + nz):] = y_part

    return data_encoding(
        det_encoding=det_encoding, err_encoding=err_encoding, i_hx=i_hx, i_hz=i_hz,
        i_hx_only=i_hx_only, i_hy_only=i_hy_only, i_hz_only=i_hz_only,
        hx_red=None, hz_red=None, hx_red_ker=None, hz_red_ker=None,
        det_index=det_index, dx=dx, dz=dz, i_dx_in_hx=i_dx_in_hx,
        i_dz_in_hz=i_dz_in_hz, big_matrix=big_matrix)


def role_encoding_for(distance, matrices, circuit, upstream_fn):
    """Return ``(encoding, source_description)``.

    ``GARI_ROLE_SOURCE`` selects the behaviour:

    ``auto`` (default)  use the layout-free labelling when it is unambiguous,
                        otherwise the upstream layout helpers;
    ``regions``         always use the layout-free labelling (ambiguous
                        detectors are then labelled 0, which the caller must
                        not rely on);
    ``layout``          always use the upstream layout helpers.
    """
    mode = os.environ.get("GARI_ROLE_SOURCE", "auto").lower()
    if mode in ("auto", "regions"):
        index, ambiguous = det_index_from_regions(circuit)
        n_one = int((index == INDEX_ONE).sum())
        n_three = int((index == INDEX_THREE).sum())
        usable = ambiguous == 0 and n_one > 0 and n_three > 0
        if mode == "regions" or usable:
            return (encode_with(matrices, circuit, index),
                    f"detecting_regions (index1={n_one} index3={n_three} "
                    f"ambiguous={ambiguous})")
    enc = upstream_fn(distance, matrices, circuit)
    return enc, f"upstream layout helpers (distance={distance})"
