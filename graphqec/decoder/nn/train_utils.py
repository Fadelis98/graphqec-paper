import os
from typing import Dict

from accelerate.utils import load_state_dict
from torch.optim.lr_scheduler import *

from graphqec.decoder.nn import QECCDecoder, get_model
from graphqec.qecc import TemporalTannerGraph

__all__ = [
    "build_neural_decoder",
    "fliter_state_dict",
    "prepend_compile_prefix",
    "remove_compile_prefix",
    ]

def build_neural_decoder(tanner_graph:TemporalTannerGraph, hyper_params:Dict[str,int]) -> QECCDecoder:
    model_name = hyper_params.pop("name")
    chkpt_path = hyper_params.pop("chkpt", None)
    
    model = get_model(
        name=model_name,
        tanner_graph=tanner_graph,
        **hyper_params
    )

    if chkpt_path is not None:
        new_state, not_matched = fliter_state_dict(
            chkpt_state_dict=load_state_dict(os.path.join(chkpt_path,"model.safetensors")),
            model_state_dict=model.state_dict(),
            compile_model = False,
            )
        if not_matched:
            print(f"Warning: {len(not_matched)} parameters are not matched in the checkpoint.")
        model.load_state_dict(new_state, strict=False)

    return model

def fliter_state_dict(chkpt_state_dict:Dict,model_state_dict:Dict, compile_model=False, verbose:bool=True):

    if compile_model:
        chkpt_state_dict = prepend_compile_prefix(chkpt_state_dict)
        model_state_dict = prepend_compile_prefix(model_state_dict)
    else:
        chkpt_state_dict = remove_compile_prefix(chkpt_state_dict)
        model_state_dict = remove_compile_prefix(model_state_dict)

    not_matched = []
    matched = []
    for k,v in model_state_dict.items():
        if k not in chkpt_state_dict:
            not_matched.append(k)
        elif v.shape != chkpt_state_dict[k].shape:
            not_matched.append(k)
        else:
            matched.append(k)
    print(f"matched keys: {len(matched)}, not matched keys: {len(not_matched)}")
    if verbose:
        print(f"not matched keys: {not_matched}")
    chkpt_state_dict = {k:v for k,v in chkpt_state_dict.items() if k not in not_matched}
    return chkpt_state_dict, not_matched

def remove_compile_prefix(state_dict:Dict):
    if next(iter(state_dict.keys())).split(".")[0] != "_orig_mod":
        return state_dict
    return {".".join(k.split(".")[1:]):v for k,v in state_dict.items()}

def prepend_compile_prefix(state_dict:Dict):
    if next(iter(state_dict.keys())).split(".")[0] == "_orig_mod":
        return state_dict
    return {".".join(["_orig_mod",k]):v for k,v in state_dict.items()}