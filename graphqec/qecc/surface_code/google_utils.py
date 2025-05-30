import stim
from typing import Optional, List, Dict, FrozenSet, Iterable, Sequence, Union, Tuple

def reverse_engineer_stim_circuit_then_split_into_two_parts(
        dem: stim.DetectorErrorModel,
        reference_circuit: stim.Circuit,
        distance: int = 3
) -> Tuple[stim.Circuit, stim.Circuit]:
    num_detectors = dem.num_detectors
    # final readout detectors
    num_readout_detectors = (distance**2-1)//2
    final_detector_start_id = num_detectors - num_readout_detectors

    final_readout_detectors = [
        stim.target_relative_detector_id(i)
        for i in range(final_detector_start_id, num_detectors)
    ]
    # flatten the circuit to avoid loop
    reference_circuit = reference_circuit.flattened()
    # select a representative error for each error mechanism in the detector error model
    representative_errors = reference_circuit.explain_detector_error_model_errors(
        dem_filter=dem,
        reduce_to_one_representative_error=True
    )
    # remove the final readout detectors from the representative errors
    representative_errors_without_final = [
        r for r in representative_errors
        if any(
            t.dem_target not in final_readout_detectors
            for t in r.dem_error_terms
            if t.dem_target.is_relative_detector_id()
        )
    ]
    representative_errors_final = [
        r for r in representative_errors
        if all(
            t.dem_target in final_readout_detectors
            for t in r.dem_error_terms
            if t.dem_target.is_relative_detector_id()
        )
    ]
    # map dem items to error rates
    dem_items_error_rates = {
        frozenset(t for t in dem_item.targets_copy() if t.is_relative_detector_id()): dem_item.args_copy()[0]
        for dem_item in dem if dem_item.type == "error"
    }

    # construct the main circuit, which is the circuit before the final readout
    final_readout_idx = _final_readout_circuit_index(reference_circuit)
    main_circuit = _replace_errors_with_representation(
        reference_circuit=reference_circuit[:final_readout_idx],
        representative_errors=representative_errors_without_final,
        error_rates=dem_items_error_rates,
    )
    # construct the final circuit
    final_reference_circuit = stim.Circuit()
    data_qubits_targets = reference_circuit[final_readout_idx].targets_copy()
    num_readouts = len(data_qubits_targets)
    final_reference_circuit.append("X_ERROR", data_qubits_targets, 0.1)
    stored_detector_instructions = []
    for ins in reference_circuit[final_readout_idx:]:
        if ins.name == "DETECTOR":
            args = [*ins.gate_args_copy()[:2], 0]
            stored_detector_instructions.append(stim.CircuitInstruction(
                "DETECTOR",
                ins.targets_copy(),
                args,
            ))
            targets = [
                t for t in ins.targets_copy()
                if t.is_measurement_record_target and t.value >= -num_readouts
            ]
            final_reference_circuit.append("DETECTOR", targets, args)
        else:
            final_reference_circuit.append(ins)

    final_dem = stim.DetectorErrorModel()
    final_dem_error_rates = dict()
    for error in representative_errors_final:
        det_targets = frozenset(t.dem_target for t in error.dem_error_terms if t.dem_target.is_relative_detector_id())
        shifted_targets = [
            stim.target_relative_detector_id(t.dem_target.val - final_detector_start_id)
            if t.dem_target.is_relative_detector_id()
            else t.dem_target
            for t in error.dem_error_terms
        ]
        shifted_det_targets = frozenset(t for t in shifted_targets if t.is_relative_detector_id())
        error_rate = dem_items_error_rates[det_targets]
        final_dem_error_rates[shifted_det_targets] = error_rate
        final_dem.append("error", (error_rate,), shifted_targets)
    representative_errors_final = final_reference_circuit.explain_detector_error_model_errors(
        dem_filter=final_dem,
        reduce_to_one_representative_error=True
    )
    final_circuit = _replace_errors_with_representation(
        reference_circuit=final_reference_circuit,
        representative_errors=representative_errors_final,
        error_rates=final_dem_error_rates,
    )
    # restore the final readout detectors
    final_circuit_restore = stim.Circuit()
    first_detector = True
    for ins in final_circuit:
        if ins.name == "DETECTOR":
            if not first_detector:
                continue
            for i in stored_detector_instructions:
                final_circuit_restore.append(i)
            first_detector = False
        else:
            final_circuit_restore.append(ins)
    return main_circuit, final_circuit_restore

def _replace_errors_with_representation(
        reference_circuit: stim.Circuit,
        representative_errors: List[stim.ExplainedError],
        error_rates: Dict[FrozenSet[stim.DemTarget], float],
) -> stim.Circuit:
    representative_errors_rates = [
        error_rates[frozenset(
            t.dem_target
            for t in representative_error.dem_error_terms
            if t.dem_target.is_relative_detector_id()
        )]
        for representative_error in representative_errors
    ]
    representative_circuit_error_locations = [r.circuit_error_locations[0] for r in representative_errors]
    representative_circuit_error_indices = [
        r.stack_frames[0].instruction_offset
        for r in representative_circuit_error_locations
    ]

    circuit = stim.Circuit()
    for i, instruction in enumerate(reference_circuit):
        if i in representative_circuit_error_indices:
            iis = [j for j, idx in enumerate(representative_circuit_error_indices) if idx == i]
            for ii in iis:
                error_rate = representative_errors_rates[ii]
                error_location = representative_circuit_error_locations[ii]
                pauli_targets = error_location.flipped_pauli_product
                if len(pauli_targets) == 1:
                    target = pauli_targets[0].gate_target
                    circuit.append("E", target, error_rate)
                elif len(pauli_targets) == 2:
                    target1 = pauli_targets[0].gate_target
                    target2 = pauli_targets[1].gate_target
                    circuit.append("E", [target1, target2], error_rate)
                else:
                    raise NotImplementedError
            continue
        if instruction.name in ["DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR"]:
            continue
        else:
            circuit.append(instruction)

    return _remove_duplicated_tick(circuit)

def _final_readout_circuit_index(
        circuit: stim.Circuit,
) -> int:
    i = -1
    while circuit[i].name != "M":
        i -= 1
    return len(circuit) + i

def _remove_duplicated_tick(
        circuit: stim.Circuit
) -> stim.Circuit:
    new_circuit = stim.Circuit()
    for ins in circuit:
        if ins.name == "TICK" and new_circuit[-1].name == "TICK":
            continue
        new_circuit.append(ins)
    return new_circuit

def final_readout_basis_change(
        circuit: stim.Circuit,
) -> List[int]:
    i = -1
    while circuit[i].name not in ["H", "SQRT_Y_DAG"]:
        i -= 1
    basis_change_ins = circuit[i]
    final_readout_ins = circuit[_final_readout_circuit_index(circuit)]
    need_basis_change = [
        i.qubit_value
        for i in basis_change_ins.targets_copy()
        if i in final_readout_ins.targets_copy()
    ]
    return need_basis_change

def prepend_basis_change(
        circuit: stim.Circuit,
        basis_change_qubits: Iterable[int],
) -> stim.Circuit:
    new_circuit = stim.Circuit()
    new_circuit.append("H", basis_change_qubits)
    new_circuit += circuit
    return new_circuit

