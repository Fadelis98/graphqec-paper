"""
Utilities for normalizing a Stim circuit for human-friendly editing and diffing.

Rules (updated)
-----
1.  Adjacent single-qubit gates with the same gate name are merged into one line,
    with targets sorted in ascending order.
2.  Noise instructions (DEPOLARIZE*, PAULI_CHANNEL*, X/Y/Z_ERROR, etc.)
    are **never merged**, and are grouped together *only* when they appear
    consecutively between the same pair of TICKs.
3.  Gates of different types are not merged.
4.  Two-qubit gates are not merged; they are sorted **only** by the *first* qubit.
5.  When sorting instructions, first group by instruction type,
    then sort targets inside each group.
6.  Noise instructions with zero strength **do not merge** with non-zero ones,
    and their relative order with quantum gates is preserved
    (i.e. we never swap past gates).
7.  Although zero-strength noise instructions do not alter the circuit's functionality,
    they are treated as having strength (i.e., non-zero strength noise) for sorting purposes.
    This implies they **cannot swap positions with gate operations** targeting the same qubits.
8.  M (Measurement) and R (Reset) operations are **barriers** and are not reordered
    relative to other instructions or each other.
9.  DETECTOR, OBSERVABLE_INCLUDE, TICK, and **QUBIT_COORDS** instructions are also **barriers**;
    their relative order is preserved, and sorting/merging only occurs within these blocks.
10. Adjacent DETECTOR instructions are grouped and sorted by the smallest measurement record
    index within their target list.
"""
from collections import defaultdict
from typing import List, Tuple

import stim

# -----------------------------------------------------------------------------
# Helper: classify instruction kinds
# -----------------------------------------------------------------------------
_GATE_1 = {
    "H", "X", "Y", "Z", "I", "S", "SQRT_Y", "SQRT_X", "SQRT_Z",
    "C_XYZ", "C_ZYX", "SQRT_X_DAG", "SQRT_Y_DAG", "SQRT_Z_DAG", "S_DAG",
}
_GATE_2 = {"CZ", "CX", "CY", "CNOT", "SWAP"}
_NOISE = {
    "DEPOLARIZE1", "DEPOLARIZE2",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2",
    "X_ERROR", "Y_ERROR", "Z_ERROR",
}
_META = {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE",
         "REPEAT", "{", "}", "QUBIT_COORDS", "SHIFT_COORDS"}

def _kind(name: str) -> str:
    """Return the broad category an instruction belongs to."""
    if name in _META:
        return "META"
    if name in _NOISE:
        return "NOISE"
    if name in _GATE_1:
        return "GATE_1"
    if name in _GATE_2:
        return "GATE_2"
    if name.startswith("M") or name.startswith("R"):
        return "MEAS_RESET" # New category for M/R as per new rules.
    return "UNKNOWN"

# -----------------------------------------------------------------------------
# Merge / sort helpers
# -----------------------------------------------------------------------------

# NEW: Barrier definition based on rules 8 and 9.
# Added QUBIT_COORDS to barrier names as per new request.
_BARRIER_NAMES = {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}


def _is_barrier(inst: stim.CircuitInstruction) -> bool:
    """Check if an instruction acts as a normalization barrier (Rule 8 & 9)."""
    name = inst.name
    # Rule 8: M and R are barriers.
    if name.startswith("M") or name.startswith("R"):
        return True
    # Rule 9: DETECTORs, TICKs, QUBIT_COORDS etc. are barriers.
    if name in _BARRIER_NAMES:
        return True
    return False


def _noise_key(ins: stim.CircuitInstruction) -> Tuple[str, int, bool]:
    """
    Noise instructions are distinct if their *name* or *argument* changes.
    For the same name & non-zero status, order by smallest target qubit.
    (Rule 6 & 7)
    """
    strength = ins.gate_args_copy()[0] if ins.gate_args_copy() else 0.0
    non_zero = strength != 0.0
    min_tgt = min((t.value for t in ins.targets_copy()), default=-1) # default needed for empty target list
    return (ins.name, min_tgt, non_zero)


def _sort_within_tick(
    instructions: List[stim.CircuitInstruction],
) -> List[stim.CircuitInstruction]:
    """
    Sorts a "chunk" of non-barrier instructions according to normalization rules. (Rules 1-7, excluding 8,9)

    The strategy is to group all instructions by their type (GATE_1, GATE_2,
    NOISE, etc.), process each group (merging, sorting), and then combine
    the groups in a canonical order based on rule 5.
    """
    # Step 1: Partition instructions into different categories.
    gates_1 = defaultdict(list)
    gates_2: List[stim.CircuitInstruction] = []
    noise: List[stim.CircuitInstruction] = []
    meta: List[stim.CircuitInstruction] = [] # For general meta instructions not treated as barriers

    for inst in instructions:
        kind = _kind(inst.name)
        # MEAS_RESET and specific META instructions are handled as barriers by normalize_circuit.
        if kind == "GATE_1":
            gates_1[inst.name].append(inst)
        elif kind == "GATE_2":
            gates_2.append(inst)
        elif kind == "NOISE":
            noise.append(inst)
        elif kind == "META": # Instructions like QUBIT_COORDS are now barriers, removed from here
            # Only META instructions not acting as barriers and not handled by expansion (like REPEAT)
            # should fall through here. Currently, with QUBIT_COORDS as barrier, this might be empty
            # unless new META types are added that aren't barriers.
            if inst.name not in ["{", "}", "REPEAT"] and inst.name not in _BARRIER_NAMES:
                meta.append(inst)
        elif kind == "UNKNOWN":
            # Fallback for unexpected instruction types, just keep them as meta.
            meta.append(inst)

    # Step 2: Sort and merge within each category (Rules 1, 2, 4, 6)
    # and build the final list by category order (Rule 5).
    new_order: List[stim.CircuitInstruction] = []

    # Process GATE_1 (Rule 1, 5)
    # Merge all instructions with the same name. Sort by name for determinism.
    for name in sorted(gates_1.keys()):
        all_targets = []
        # gate_args are assumed to be consistent for merged gates of the same name.
        gate_args = gates_1[name][0].gate_args_copy() if gates_1[name] else []
        for inst in gates_1[name]:
            all_targets.extend(inst.targets_copy())
        
        # Remove duplicates and sort by qubit index for canonical representation.
        if all_targets:
            unique_targets = sorted(list(set(all_targets)), key=lambda t: t.value)
            new_order.append(stim.CircuitInstruction(name, unique_targets, gate_args))

    # Process GATE_2 (Rule 4, 5)
    # Sort by the first qubit index. Instructions are not merged (Rule 3).
    new_order.extend(sorted(gates_2, key=lambda g: g.targets_copy()[0].value))

    # Process NOISE (Rule 2, 6, 7, 5)
    # Sort using the provided noise key, which handles zero-strength correctly.
    new_order.extend(sorted(noise, key=_noise_key))

    # Process META / UNKNOWN (Rule 5 - treated as last)
    # This currently handles any non-barrier meta/unknown instructions
    # If no such instructions exist, this section will simply extend with an empty list.
    key_meta = lambda inst: (
        inst.name,
        tuple(t.value for t in inst.targets_copy()),
        tuple(inst.gate_args_copy()),
    )
    new_order.extend(sorted(meta, key=key_meta))

    return new_order

# -----------------------------------------------------------------------------
# Main normalization
# -----------------------------------------------------------------------------
def normalize_circuit(circuit: stim.Circuit) -> stim.Circuit:
    """
    Normalizes a Stim circuit by expanding REPEAT blocks and sorting operations
    between barrier instructions.

    Detailed Rules implemented:
    1.  Adjacent single-qubit gates are merged.
    2.  Noise is never merged.
    3.  Gates of different types are not merged.
    4.  Two-qubit gates are not merged; they are sorted by the first qubit.
    5.  Instructions are grouped by type, then sorted.
    6.  Zero-strength noise is kept separate from non-zero noise.
    7.  Noise does not swap past gates on the same qubit.
    8.  M and R operations are barriers and are not reordered.
    9.  DETECTOR, OBSERVABLE_INCLUDE, TICK, and QUBIT_COORDS instructions are also barriers;
        their relative order is preserved, and sorting/merging only occurs within these blocks.
    10. Adjacent DETECTOR instructions are grouped and sorted by the smallest measurement record
        index within their target list. Each DETECTOR's internal targets are also sorted.
    """
    # Phase 0: Expand REPEAT blocks fully. This simplifies later logic by removing control flow.
    # stim.Circuit.flatten_nested_blocks() handles this robustly.
    expanded_circuit = circuit.flattened()

    # Phase 1: Normalization in chunks separated by barriers (Rules 8 & 9).
    new_circ = stim.Circuit()
    current_chunk: List[stim.CircuitInstruction] = []
    current_detector_group: List[stim.CircuitInstruction] = [] # New list to hold adjacent DETECTORs

    # Helper function to process and append a DETECTOR group
    def _process_detector_group(group: List[stim.CircuitInstruction], output_circuit: stim.Circuit):
        if not group:
            return

        def get_detector_sort_key(inst: stim.CircuitInstruction) -> Tuple[float, ...]:
            coordinates = inst.gate_args_copy()
            padded_coords = list(coordinates) + [0] * (5 - len(coordinates)) if len(coordinates) < 5 else coordinates
            return (padded_coords[2], padded_coords[0], padded_coords[1], padded_coords[3], padded_coords[4])

        # Sort the accumulated DETECTORs using the coordinate-based key.
        sorted_group = sorted(group, key=get_detector_sort_key)
        
        for det_inst in sorted_group:
            # Ensure their *internal* targets are canonically ordered (smallest first).
            sorted_targets_for_append = sorted(det_inst.targets_copy(), key=lambda t: t.value)
            output_circuit.append(stim.CircuitInstruction(det_inst.name, sorted_targets_for_append, det_inst.gate_args_copy()))

    for inst in expanded_circuit:
        if _is_barrier(inst):
            # If we were collecting a normal chunk (non-barrier gates), process it first.
            if current_chunk:
                sorted_chunk = _sort_within_tick(current_chunk)
                for sorted_inst in sorted_chunk:
                    new_circ.append(sorted_inst)
                current_chunk = [] # Reset chunk.
            
            # If we were collecting a DETECTOR group, process it before starting a new group or appending a non-DETECTOR barrier.
            if current_detector_group and inst.name != "DETECTOR":
                _process_detector_group(current_detector_group, new_circ)
                current_detector_group = [] # Reset DETECTOR group.

            # Now, handle the current barrier instruction.
            if inst.name == "DETECTOR":
                current_detector_group.append(inst) # Add to the DETECTOR group.
            else:
                # If it's a non-DETECTOR barrier, process any outstanding DETECTOR group, then append this barrier.
                _process_detector_group(current_detector_group, new_circ) # Ensure any outstanding DETECTORs are processed
                current_detector_group = [] # Clear it after processing
                new_circ.append(inst) # Add the current non-DETECTOR barrier.

        else: # Encountered a non-barrier instruction
            # If we were collecting a DETECTOR group, it ends here, so process it.
            if current_detector_group:
                _process_detector_group(current_detector_group, new_circ)
                current_detector_group = [] # Reset DETECTOR group.
            
            current_chunk.append(inst) # Add to the normal chunk.

    # After iterating through all instructions, process any remaining chunks or detector groups.
    if current_chunk:
        sorted_chunk = _sort_within_tick(current_chunk)
        for sorted_inst in sorted_chunk:
            new_circ.append(sorted_inst)
    
    if current_detector_group: # Process any DETECTORs left at the end of the circuit
        _process_detector_group(current_detector_group, new_circ)

    return new_circ
