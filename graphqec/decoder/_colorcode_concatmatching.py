"""modified from https://github.com/seokhyung-lee/color-code-stim"""

from typing import Dict, Iterable, List, Literal, Set, Tuple

import numpy as np
import pymatching
import stim


class ErrorMechanismSymbolic:
    prob_vars: np.ndarray
    prob_muls: np.ndarray
    dets: Set[stim.DemTarget]
    obss: Set[stim.DemTarget]

    def __init__(
        self,
        prob_vars: Iterable[int],
        dets: Iterable[stim.DemTarget],
        obss: Iterable[stim.DemTarget],
        prob_muls: Iterable[float] | int | float = 1,
    ):
        prob_vars = np.asarray(prob_vars, dtype="int32")
        prob_muls = np.asarray(prob_muls)
        if prob_muls.ndim == 0:
            prob_muls = np.full_like(prob_vars, prob_muls)

        self.prob_vars = prob_vars
        self.prob_muls = prob_muls
        self.dets = set(dets)
        self.obss = set(obss)


class DemSymbolic:
    ems: List[ErrorMechanismSymbolic]
    dets_orig: stim.DetectorErrorModel

    def __init__(
        self,
        prob_vars: Iterable[Iterable[int]],
        dets: Iterable[Iterable[stim.DemTarget]],
        obss: Iterable[Iterable[stim.DemTarget]],
        dets_orig: stim.DetectorErrorModel,
    ):
        self.ems = [
            ErrorMechanismSymbolic(*prms) for prms in zip(prob_vars, dets, obss)
        ]
        self.dets_orig = dets_orig

    def to_dem(self, prob_vals: Iterable[float], sort=False):
        prob_vals = np.asarray(prob_vals, dtype="float64")

        probs = [
            (1 - np.prod(1 - 2 * em.prob_muls * prob_vals[em.prob_vars])) / 2
            for em in self.ems
        ]

        if sort:
            inds = np.argsort(probs)[::-1]
        else:
            inds = range(len(probs))

        dem = stim.DetectorErrorModel()
        for i in inds:
            em = self.ems[i]
            targets = em.dets | em.obss
            dem.append("error", probs[i], list(targets))

        dem += self.dets_orig

        return dem

    def non_edge_like_errors_exist(self):
        for e in self.ems:
            if len(e.dets) > 2:
                return True
        return False

    def decompose_complex_error_mechanisms(self):
        """
        For each error mechanism `e` in `dem` that involves more than two detectors,
        searches for candidate pairs (e1, e2) among the other error mechanisms (with e1, e2 disjoint)
        such that:
        - e1.dets ∪ e2.dets equals e.dets, and e1.dets ∩ e2.dets is empty.
        - e1.obss ∪ e2.obss equals e.obss, and e1.obss ∩ e2.obss is empty.

        For each valid candidate pair, updates both e1 and e2 by concatenating e’s
        probability variable and multiplier arrays. If there are multiple candidate pairs,
        the probability multipliers from e are split equally among the pairs.

        Finally, removes the complex error mechanism `e` from `dem.ems`.

        Raises:
            ValueError: If a complex error mechanism cannot be decomposed.
        """
        # Iterate over a copy of the error mechanisms list
        em_inds_to_remove = []
        for i_e, e in enumerate(self.ems):
            # Process only error mechanisms that involve more than 2 detectors.
            if len(e.dets) > 2:
                candidate_pairs = []
                # Search for candidate pairs among the other error mechanisms.
                for i, e1 in enumerate(self.ems):
                    if i == i_e or i in em_inds_to_remove:
                        continue
                    for j in range(i + 1, len(self.ems)):
                        if j == i_e or j in em_inds_to_remove:
                            continue
                        e2 = self.ems[j]
                        # Check that e1 and e2 have disjoint detectors and observables.
                        if e1.dets & e2.dets:
                            continue
                        if e1.obss & e2.obss:
                            continue
                        # Check that the union of their detectors and observables equals e’s.
                        if (e1.dets | e2.dets == e.dets) and (
                            e1.obss | e2.obss == e.obss
                        ):
                            candidate_pairs.append((e1, e2))
                if not candidate_pairs:
                    raise ValueError(
                        f"No valid decomposition found for error mechanism with dets {e.dets} and obss {e.obss}."
                    )
                # If there are multiple decompositions, split the probability equally.
                fraction = 1 / len(candidate_pairs)
                for e1, e2 in candidate_pairs:
                    # Append the probability variable arrays.
                    e1.prob_vars = np.concatenate([e1.prob_vars, e.prob_vars])
                    e2.prob_vars = np.concatenate([e2.prob_vars, e.prob_vars])
                    # Append the probability multiplier arrays, scaling by the fraction.
                    e1.prob_muls = np.concatenate(
                        [e1.prob_muls, e.prob_muls * fraction]
                    )
                    e2.prob_muls = np.concatenate(
                        [e2.prob_muls, e.prob_muls * fraction]
                    )
                # Remove the complex error mechanism from the model.
                em_inds_to_remove.append(i_e)

            for i_e in em_inds_to_remove[::-1]:
                self.ems.pop(i_e)


class ColorCodeDecoder:

    def __init__(
            self,
            dem: stim.DetectorErrorModel,
            detector_colors,
            detector_basis,
            logical_basis = "Z",
            verbose: bool = False,
            ):
        """
        Initializes the ColorCodeDecoder class.

        Args:
            dem (stim.DetectorErrorModel): The original detector error model.
            detector_colors: Array or list specifying the color of each detector.
            detector_basis: Array or list specifying the basis (X or Z) of each detector.
            logical_basis (Literal["Z", "X"], optional): The logical basis. Defaults to "Z".
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """

        self.verbose = verbose
        self.color_map = {'r':0, 'g':1, 'b':2}

        self.dems: Dict[
            Literal["r","g","b",],
            Tuple[stim.DetectorErrorModel, stim.DetectorErrorModel]
        ]

        dems = {}
        color_masks = {}
        for color in ["r", "g", "b"]:
            mask = np.array(detector_colors)==self.color_map[color]
            color_masks[color] = mask
            dem1, dem2 = self.decompose_detector_error_model(
                dem,
                color,
                detector_colors,
                detector_basis,
                logical_basis,
                )
            dems[color] = dem1, dem2  # stim.DetectorErrorModel

        self.dems = dems
        self.color_masks = color_masks

    def decompose_detector_error_model(
        self,
        orig_dem:stim.DetectorErrorModel,
        color,
        detector_colors,
        detector_basis,
        logical_basis: Literal["Z","X"] = "Z",
        decompose_non_edge_like_errors: bool = True,
    ) -> Tuple[stim.DetectorErrorModel, stim.DetectorErrorModel]:
        """
        Decomposes the detector error model (DEM) into restricted and monochromatic DEMs for a given color.

        Args:
            orig_dem (stim.DetectorErrorModel): The original detector error model.
            color (Literal["r", "g", "b"]): The color for decomposition.
            detector_colors: Array or list specifying the color of each detector.
            detector_basis: Array or list specifying the basis (X or Z) of each detector.
            logical_basis (Literal["Z", "X"], optional): The logical basis. Defaults to "Z".
            decompose_non_edge_like_errors (bool, optional): Whether to remove non-edge-like error mechanisms. Defaults to True.

        Returns:
            Tuple[stim.DetectorErrorModel, stim.DetectorErrorModel]: 
                - dem1: The restricted DEM for the given color.
                - dem2: The monochromatic DEM for the given color.
        """

        if isinstance(detector_colors, List):
            detector_colors = np.array(detector_colors)
        if isinstance(detector_basis, List):
            detector_basis = np.array(detector_basis)

        # Set of detector ids to be reduced
        color_map = {'r':0, 'g':1, 'b':2}
        det_ids_to_reduce = set(np.where(detector_colors==color_map[color])[0])

        # Original DEM
        # orig_dem = self.orig_dem
        # num_orig_error_sources = orig_dem.num_errors
        num_detectors = orig_dem.num_detectors
        # orig_dem_dets = orig_dem[num_orig_error_sources:]
        # orig_dem_errors = orig_dem[:num_orig_error_sources]
        orig_dem_dets = stim.DetectorErrorModel()
        orig_dem_errors = stim.DetectorErrorModel()
        for inst in orig_dem.flattened():
            if inst.type == 'error':
                orig_dem_errors.append(inst)
            if inst.type == 'detector':
                orig_dem_dets.append(inst)
        
        orig_probs = np.array(
            [inst.args_copy()[0] for inst in orig_dem_errors], dtype="float64"
        )

        # Decompose into X and Z errors
        pauli_decomposed_targets_dict = {}
        pauli_decomposed_probs_dict = {}

        for i_inst, inst in enumerate(orig_dem_errors):
            targets = inst.targets_copy()
            new_targets = {"Z": [], "X": []}
            new_target_ids = {"Z": set(), "X": set()}

            for target in targets:
                if target.is_logical_observable_id():
                    obsid = int(str(target)[1:])
                    pauli = logical_basis
                    new_targets[pauli].append(target)
                    new_target_ids[pauli].add(f"L{obsid}")
                else:
                    detid = int(str(target)[1:])

                    pauli = detector_basis[detid]

                    new_targets[pauli].append(target)
                    new_target_ids[pauli].add(detid)

            for pauli in ["Z", "X"]:
                new_targets_pauli = new_targets[pauli]
                if new_targets_pauli:
                    new_target_ids_pauli = frozenset(new_target_ids[pauli])
                    try:
                        pauli_decomposed_probs_dict[new_target_ids_pauli].append(i_inst)
                    except KeyError:
                        pauli_decomposed_probs_dict[new_target_ids_pauli] = [i_inst]
                        pauli_decomposed_targets_dict[new_target_ids_pauli] = (
                            new_targets_pauli
                        )

        # Obtain targets list for the two steps
        dem1_probs_dict = {}
        dem1_dets_dict = {}
        dem1_obss_dict = {}
        dem1_virtual_obs_dict = {}

        dem2_probs = []
        dem2_dets = []
        dem2_obss = []

        for target_ids in pauli_decomposed_targets_dict:
            targets = pauli_decomposed_targets_dict[target_ids]
            prob = pauli_decomposed_probs_dict[target_ids]

            dem1_dets_sng = []
            dem1_obss_sng = []
            dem2_dets_sng = []
            dem2_obss_sng = []
            dem1_det_ids = set()

            for target in targets:
                if target.is_logical_observable_id():
                    dem2_obss_sng.append(target)
                else:
                    det_id = int(str(target)[1:])
                    if det_id in det_ids_to_reduce:
                        dem2_dets_sng.append(target)
                    else:
                        dem1_dets_sng.append(target)
                        dem1_det_ids.add(det_id)

            if not decompose_non_edge_like_errors:
                if dem1_dets_sng:
                    if len(dem1_dets_sng) >= 3 or len(dem2_dets_sng) >= 2:
                        continue
                else:
                    if len(dem2_dets_sng) >= 3:
                        continue

            if dem1_det_ids:
                dem1_det_ids = frozenset(dem1_det_ids)
                try:
                    dem1_probs_dict[dem1_det_ids].extend(prob)
                    virtual_obs = dem1_virtual_obs_dict[dem1_det_ids]
                except KeyError:
                    virtual_obs = len(dem1_probs_dict)
                    dem1_obss_sng.append(stim.target_logical_observable_id(virtual_obs))
                    dem1_probs_dict[dem1_det_ids] = prob
                    dem1_dets_dict[dem1_det_ids] = dem1_dets_sng
                    dem1_obss_dict[dem1_det_ids] = dem1_obss_sng
                    dem1_virtual_obs_dict[dem1_det_ids] = virtual_obs

                virtual_det_id = num_detectors + virtual_obs
                dem2_dets_sng.append(stim.target_relative_detector_id(virtual_det_id))

            dem2_dets.append(dem2_dets_sng)
            dem2_obss.append(dem2_obss_sng)
            dem2_probs.append(prob)

        # Convert dem1 information to lists
        dem1_probs = list(dem1_probs_dict.values())
        dem1_dets = [dem1_dets_dict[key] for key in dem1_probs_dict]
        dem1_obss = [dem1_obss_dict[key] for key in dem1_probs_dict]

        # Convert to DemTuple objects
        dem1_sym = DemSymbolic(dem1_probs, dem1_dets, dem1_obss, orig_dem_dets)
        dem2_sym = DemSymbolic(dem2_probs, dem2_dets, dem2_obss, orig_dem_dets)

        dem1 = dem1_sym.to_dem(orig_probs)
        dem2 = dem2_sym.to_dem(orig_probs, sort=True)

        return dem1, dem2

    def decode(self,syndromes: np.ndarray):
        """
        Decodes given detector outcomes using the decomposed DEMs.

        Args:
            syndromes (np.ndarray): A 2D array of detector outcomes. Each row corresponds to a sample,
                and each column corresponds to a detector. syndromes[i, j] is True if the detector
                with id j in the ith sample has the outcome -1.

        Returns:
            np.ndarray: Predicted observables. It is 1D if there is only one observable and 2D otherwise.
                preds_obs[i] or preds_obs[i, j] is True if the j-th observable (j=0 when 1D) of the i-th
                sample is predicted to be -1.
        """

        preds_obs = None
        weights = None
        all_colors = {}
        for c, dems_color in self.dems.items():
            dem1, dem2 = dems_color


            if self.verbose:
                print(f"color {c}, step-1 decoding..")
            preds_dem1 = self._decode_dem1(dem1, syndromes, self.color_masks[c])
            if self.verbose:
                print(f"color {c}, step-2 decoding..")
            preds_obs_new, weights_new = self._decode_dem2(
                dem2, syndromes, preds_dem1, self.color_masks[c]
            )
            del preds_dem1

            if self.verbose:
                print(f"color {c}, postprocessing..")

            all_colors[c] = (preds_obs_new, weights_new)

            if preds_obs is None:
                preds_obs = preds_obs_new
                weights = weights_new
            else:
                cond = weights_new < weights
                preds_obs = np.where(cond.reshape(-1, 1), preds_obs_new, preds_obs)
                weights = np.where(cond, weights_new, weights)

        preds_obs = preds_obs.astype("bool")

        return preds_obs

    def _decode_dem1(self, dem1, syndromes, color_mask):
        det_outcomes_dem1 = syndromes.copy()
        det_outcomes_dem1[:, color_mask] = False
        matching = pymatching.Matching.from_detector_error_model(dem1)
        preds_dem1 = matching.decode_batch(det_outcomes_dem1)
        del det_outcomes_dem1, matching

        return preds_dem1

    def _decode_dem2(self, dem2, syndromes, preds_dem1, color_mask):
        det_outcome_dem2 = syndromes.copy()
        mask = np.full_like(det_outcome_dem2, True)
        mask[:, color_mask] = False
        det_outcome_dem2[mask] = False
        del mask
        det_outcome_dem2 = np.concatenate([det_outcome_dem2, preds_dem1], axis=1)
        matching = pymatching.Matching.from_detector_error_model(dem2)
        preds, weights_new = matching.decode_batch(
            det_outcome_dem2, return_weights=True
        )
        del det_outcome_dem2, matching
        return preds, weights_new