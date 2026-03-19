import os
from typing import Dict, List

import torch
from accelerate import Accelerator
from accelerate.utils import load_state_dict
from torch.optim.lr_scheduler import *
from torch.optim.optimizer import Optimizer

from graphqec.decoder.nn import QECCDecoder, get_model
from graphqec.decoder.nn.dataloader import *
from graphqec.qecc import QuantumCode, TemporalTannerGraph

__all__ = [
    "get_dataloaders",
    "build_neural_decoder",
    "AdamWReletiveWeightDecayHook",
    "fliter_state_dict",
    "prepend_compile_prefix",
    "remove_compile_prefix",
    "construct_param_groups",
    'get_optimizer',
    "construct_annealing_scheduler",
    ]
        
def get_dataloaders(quantum_code:QuantumCode, hyper_params:Dict[str,int], accelerator: Accelerator = None):
    
    data_type = hyper_params["dataloader"].get("type",'incremental')
    
    if data_type == 'incremental':
        noise_args = {
        "physical_error_rate":hyper_params["dataloader"].get("physical_error_rate",None), 
        "parity": hyper_params["dataloader"].get('parity', None)
        }

        offset = hyper_params["dataloader"].get("offset",0)
        cycle_step = hyper_params["dataloader"].get("cycle_step",1)


        train_loader = get_incremental_dataloader(
            quantum_code,
            max_num_cycle = hyper_params["dataloader"]['max_num_cycle'],
            noise_args    = noise_args,
            batch_size    = hyper_params["dataloader"]["batch_size"],
            num_samples   = hyper_params["dataloader"]['num_train'],
            num_workers   = hyper_params["dataloader"]["num_workers"],
            shuffle       = hyper_params["dataloader"]["shuffle"],
            seed          = hyper_params["training"]['seed'],
            offset        = offset,
            cycle_step    = cycle_step
            )

        val_loader = get_incremental_dataloader(
            quantum_code,
            max_num_cycle = hyper_params["dataloader"]['max_num_cycle'],
            noise_args    = noise_args,
            batch_size    = hyper_params["dataloader"]["batch_size"],
            num_samples = hyper_params["dataloader"]['num_validation'],
            num_workers   = hyper_params["dataloader"]["num_workers"],
            shuffle       = hyper_params["dataloader"]["shuffle"],
            seed          = hyper_params["training"]['seed'],
            offset        = offset + hyper_params["dataloader"]['num_train'],
            cycle_step    = cycle_step
            )
        
        if accelerator is not None:
            train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
        
        return train_loader, val_loader
    
    elif data_type in ['exp','sim']:
        from graphqec.decoder.nn.trainer import CurriculumTeacher

        noise_args = {
            "parity": hyper_params["dataloader"].get('parity', 0)
            }

        if data_type == 'exp':
            train_loaders,val_loaders = get_exp_dataloaders(
                quantum_code,
                test_cycles = range(0,hyper_params["dataloader"].get('max_num_cycle',25),2), # NOTE hard-coding for sycamore data now
                noise_args=noise_args,
                batch_size  = hyper_params["dataloader"]["batch_size"],
                val_batch_size = hyper_params["dataloader"]["num_validation"],
                num_samples = hyper_params["dataloader"]['num_train'],
                num_workers = hyper_params["dataloader"]["num_workers"],
                shuffle     = hyper_params["dataloader"]["shuffle"],
                seed=hyper_params["training"]['seed'],
                )
        if data_type == 'sim':
            train_loaders = get_sim_dataloaders(
                quantum_code,
                test_cycles = range(0,hyper_params["dataloader"].get('max_num_cycle',25),2), # NOTE hard-coding for zuchongzhi data now
                noise_args=noise_args,
                batch_size  = hyper_params["dataloader"]["batch_size"],
                num_samples = hyper_params["dataloader"]['num_train'],
                num_workers = hyper_params["dataloader"]["num_workers"],
                shuffle     = hyper_params["dataloader"]["shuffle"],
                seed=hyper_params["training"]['seed'],
                )
            val_loaders = get_sim_dataloaders(
                quantum_code,
                test_cycles = range(0,hyper_params["dataloader"].get('max_num_cycle',25),2), # NOTE hard-coding for zuchongzhi data now
                noise_args=noise_args,
                batch_size  = hyper_params["dataloader"]["batch_size"],
                num_samples = hyper_params["dataloader"]['num_validation'],
                num_workers = hyper_params["dataloader"]["num_workers"],
                shuffle     = hyper_params["dataloader"]["shuffle"],
                seed=hyper_params["training"]['seed'],
                )


        if accelerator is not None:
            train_loaders = [accelerator.prepare(loader) for loader in train_loaders]
            val_loaders   = [accelerator.prepare(loader) for loader in val_loaders]

        train_teacher = CurriculumTeacher(train_loaders, seed=hyper_params["training"]['seed'])
        val_teacher   = CurriculumTeacher(val_loaders, seed=hyper_params["training"]['seed'])
        if accelerator is not None:
            accelerator.register_for_checkpointing(train_teacher)
            accelerator.register_for_checkpointing(val_teacher)

        return train_teacher, val_teacher

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

class AdamWReletiveWeightDecayHook:
    """decay the parameters with respect to their initial values"""

    def __init__(self, param_groups, weight_decay=0.01, device = None) -> None:
        self.pretrain_param_groups = [{"params":[torch.zeros_like(p,device=device).copy_(p) for p in group["params"]]} for group in param_groups]
       
        self.weight_decay = weight_decay
        assert weight_decay > 0, "weight decay must be greater than 0"

    def __call__(self, optimizer, *args, **kwargs) -> None:
        assert len(optimizer.param_groups) == len(self.pretrain_param_groups)
        for opt_group, pretrain_group in zip(optimizer.param_groups, self.pretrain_param_groups):
            if opt_group["weight_decay"] > 0:
                # if the weight decay is already set, skip this step
                continue
            params = opt_group["params"]
            pretrain_params = pretrain_group["params"]
            lr = opt_group["lr"]
            assert len(params) == len(pretrain_params), "the number of parameters is not equal"
            with torch.no_grad():
                for i, p in enumerate(params):
                    p: torch.nn.Parameter
                    if p.requires_grad:
                        # compute the difference between the current parameter and the pretrained parameter
                        delta_p = p - pretrain_params[i]
                        decay_factor = lr * self.weight_decay
                        p.add_(delta_p, alpha=-decay_factor)

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

def construct_param_groups(model:torch.nn.Module, not_matched:List[str], adapt_lr: float = None):
        if len(not_matched) > 0 and adapt_lr is not None:
            finetune_params = []
            new_params = []
            for n,p in model.named_parameters():
                if n not in not_matched:
                    finetune_params.append(p)
                else:
                    new_params.append(p)
            if adapt_lr == 0: 
                # frozen pretrained params
                param_group_strategy = "frozen"
                param_groups = [
                    {
                        # use the default lr and weight decay for new params
                        'params': new_params, 
                    }]
                for p in finetune_params:
                    p.requires_grad_(False)               
            else:
                param_group_strategy = "adaptive"
                param_groups = [
                    {
                        'params': new_params,
                    },
                    {
                        'params': finetune_params,
                        'lr': adapt_lr,
                    }]
        else:
            param_group_strategy = "full"
            param_groups = [{
                'params': list(model.parameters()),
            }]

        return param_groups, param_group_strategy

def get_optimizer(param_groups:List[Dict],
                  hyper_params:Dict,
                  device: torch.device | str,
                  ):

    optimizer_name = hyper_params['training'].get("optimizer", "adamw")
    realative_weight_decay = hyper_params['training'].get("relative_weight_decay",False)

    if optimizer_name == "lamb":
        from pytorch_optimizer import Lamb
        optimizer_cls = Lamb
    elif optimizer_name == "adamw":
        from torch.optim import AdamW
        optimizer_cls = AdamW
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")

    betas = (hyper_params["training"].get('beta1', 0.99), hyper_params["training"].get('beta2', 0.9999))
    lr = hyper_params["training"]['lr']
    weight_decay = hyper_params["training"].get('weight_decay',0.01)

    if realative_weight_decay:
        weight_decay_hook = AdamWReletiveWeightDecayHook(param_groups,weight_decay,device)
        weight_decay = 0.0

    optimizer = optimizer_cls(param_groups, 
                    betas = betas,
                    lr=lr,
                    weight_decay=weight_decay)
    
    if realative_weight_decay:
        optimizer.register_step_post_hook(weight_decay_hook)

    return optimizer

class IDLEScheduler(LRScheduler):
    """A scheduler keeps the learning rate constant"""
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = 'deprecated') -> None:
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def construct_annealing_scheduler(optimizer, num_warmup_steps, num_annealing_steps, final_lr, num_adapt_steps = None):
    warmupLR = LinearLR(optimizer,1e-7, 1, num_warmup_steps)
    cosineLR = CosineAnnealingLR(optimizer, num_annealing_steps, final_lr)
    constantLR = IDLEScheduler(optimizer)

    scheduler_list  = [warmupLR, cosineLR, constantLR]
    milestones = [num_warmup_steps,num_warmup_steps+num_annealing_steps]

    scheduler = SequentialLR(optimizer, scheduler_list, milestones)
    return scheduler