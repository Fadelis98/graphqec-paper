import argparse
import json
import os
import socket
from typing import Dict

import accelerate
import torch
from accelerate import Accelerator
from accelerate.utils import load_state_dict, set_seed
from torch.optim.lr_scheduler import *

from graphqec.decoder.nn.profiler import count_parameters, generate_parameter_table
from graphqec.decoder.nn.train_utils import *
from graphqec.decoder.nn.trainer import *
from graphqec.qecc import get_code

# Global flags
DEBUG = False
COMPILE_MODEL = False
NO_RESUME = False
NO_CUDA = False
SAVE_PROFILER_RESULTS = False
STAT_GRAD_NORM = False
USE_FSDP = False
NO_TRACKER = False
TRACKER = 'tensorboard'

# enable TF32
torch.backends.cuda.matmul.allow_tf32 = True

def main(accelerator: Accelerator,
         checkpointor: CheckPointor,
         hyper_params: Dict,
        ):
 
    # initialize training objects
    set_seed(hyper_params["training"]['seed'])
    test_code = get_code(hyper_params["code"]['code_type'], **hyper_params["code"])

    # get dataloaders
    train_loader, val_loader = get_dataloaders(test_code, hyper_params, accelerator)
    epoch_step_iterator = EpochStepIterator(train_loader,hyper_params["training"]['num_steps'],hyper_params["training"].get('num_epochs',1),verbose=True)
    accelerator.register_for_checkpointing(epoch_step_iterator)

    # build model with tanner graph
    tanner_graph = test_code.get_tanner_graph().to(accelerator.device)
    cycle_step = hyper_params['dataloader'].get('cycle_step',1)
    if hyper_params['dataloader'].get('type','incremental') == 'incremental':
        hyper_params['model']['incremental_step'] = cycle_step
    else:
        hyper_params['model']['incremental_step'] = None
    model = build_neural_decoder(tanner_graph, hyper_params["model"])

    # profile model parameters
    if accelerator.is_main_process:
        total_params, param_summary, submodule_details = count_parameters(model, depth=1)
        accelerator.print(generate_parameter_table(submodule_details))
        if TRACKER == "wandb" and not NO_TRACKER:
            accelerator.get_tracker('wandb').store_init_configuration({"num_params": total_params})

    # compile
    if COMPILE_MODEL:
        model = torch.compile(model, mode="reduce-overhead")

    # if chkpt is given, initialize the model from checkpoint
    # chkpt_path = hyper_params["finetuning"]["checkpoint"]
    chkpt_path: str = hyper_params.setdefault("finetuning",{}).get("checkpoint",None)
    if chkpt_path is not None:
        if chkpt_path.startswith("./"):
            chkpt_path = os.path.join(checkpointor.checkpoint_dir,chkpt_path[2:])
            accelerator.print(f"{chkpt_path} is relative path, using {chkpt_path}")

        if not os.path.exists(chkpt_path):
            raise ValueError(f"{chkpt_path} does not exist")

        new_state, not_matched = fliter_state_dict(
            chkpt_state_dict=load_state_dict(os.path.join(chkpt_path,"model.safetensors")),
            model_state_dict=model.state_dict(),
            compile_model = COMPILE_MODEL,
            )
        model.load_state_dict(new_state, strict=False)
        accelerator.print(f"Finetuning from checkpoint:{chkpt_path}")
        finetunning = True
    else:
        accelerator.print(f"Training from scratch")
        not_matched = []
        finetunning = False

    # build optimizer
    param_groups, param_group_strategy = construct_param_groups(model,not_matched,hyper_params["finetuning"].get("adapt_lr",None))
    optimizer = get_optimizer(param_groups, hyper_params, accelerator.device)
    accelerator.print(f"optimizer initialized with strategy: {param_group_strategy}")

    # load optimizer state
    if finetunning and hyper_params["finetuning"].get("load_optimizer_state",False):
        try:
            accelerator.print(f"Trying to load optimizer state")
            optimizer_state = torch.load(os.path.join(chkpt_path,"optimizer.bin"),weights_only=True)
            # assert optimizer.state_dict() == optimizer_state
            optimizer.state = optimizer_state['state']
        except Exception as e:
            print(f"Error loading optimizer state: {e}")

    # build scheduler
    scheduler = construct_annealing_scheduler(
        optimizer,
        hyper_params["training"]['warmup_steps'],
        hyper_params["training"]['annealing_steps'],
        hyper_params["training"]['final_lr'],
        )
    
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # load checkpoint
    if checkpointor.has_checkpoint:
        checkpointor.load()
    
    trainer = QECDecoderTrainer(
        model, optimizer, scheduler, accelerator, 
        loss_fn=hyper_params['training']['loss_fn'],
        clip_grad_norm=hyper_params['training']['norm_clip'],
        mask_abnormal=hyper_params['training']['mask_abnormal'],
        log_interval=hyper_params['training']['log_interval'],
        stat_grad_norm = STAT_GRAD_NORM,
        )

    last_best = float('inf')
    last_best_lfr = float('inf')
    early_stop_count = 0
    early_stop_patience = hyper_params['training'].get('early_stop_patience', -1)
    early_stop_warmup = hyper_params['training'].get('early_stop_warmup', -1)

    enable_early_stopping = early_stop_patience > 0

    with accelerator.profile() as prof:
        for current_step, current_epoch, (syndromes, obs_flips) in epoch_step_iterator:
            with accelerator.accumulate(model):
                trainer.train_batch(syndromes, obs_flips, current_step=current_step)
                prof.step()
                if accelerator.gradient_state.sync_gradients and current_step % hyper_params["training"]['log_interval'] == 0:
                    checkpointor.save(current_step)
                    f0, lfr, lfrstd = trainer.benchmark_model(val_loader, current_step, cycle_step=cycle_step)
                    val_loss = trainer.last_val_loss
                    accelerator.print(
                        f"step {current_step}, val loss {val_loss:.4e}, fitted lfr: {lfr:.4%} ± {lfrstd:.5%}, f0={f0:.3f}"
                    )

                    if val_loss < last_best:
                        last_best = val_loss
                        early_stop_count = 0

                        if enable_early_stopping:
                            checkpointor.save("val_best")
                    else:
                        if enable_early_stopping and current_step >= early_stop_warmup:
                            early_stop_count += 1

                    if lfr < last_best_lfr:
                        last_best_lfr = lfr
                        if enable_early_stopping:
                            checkpointor.save("lfr_best")

                    if enable_early_stopping and early_stop_patience > 0 and current_step >= early_stop_warmup and early_stop_count >= early_stop_patience:
                        accelerator.print("Early stopping triggered. Training terminated.")
                        break

    # end the training
    accelerator.end_training()

def parse_args():
    # Set up argument parser with clear help messages
    parser = argparse.ArgumentParser(description="Training script for decoder model")
    
    # Path and configuration arguments
    parser.add_argument('--run_dir', type=str, default='runs', help='Path to save runs (default: runs)')
    parser.add_argument('--config_file', type=str, required=True, help='Path to config file')
    
    # Experiment identification arguments
    parser.add_argument('--project_name', type=str, default='graphqec', help='Tracker project name')
    parser.add_argument('--experiment_name', type=str, default=None,help='Experiment name (defaults to config filename)')
    parser.add_argument('--experiment_group', type=str, default=None, help='Experiment group name (defaults to first part of experiment name)')
    parser.add_argument('--experiment_stage', type=str, default="pretrain", help='Experiment stage (default: pretrain)')
    
    # Feature flags and debugging options
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode, many hyper parameters will be modified')
    parser.add_argument('--no_track_experiment', action='store_true', default=False, help='Do not track experiment')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Do not use cuda')
    parser.add_argument('--no_resume', action='store_true', default=False, help='Run a new experiment anyway, only work when debugging')
    parser.add_argument('--trace_profiler', action='store_true', default=False, help='Trace profiler for debugging')
    parser.add_argument('--stat_grad_norm', action='store_true', default=False, help='Log gradient norm in training')
    
    # Technical options
    parser.add_argument('--tracker', type=str, choices=['wandb', 'tensorboard'], default='tensorboard', help='Experiment tracker (default: tensorboard)')
    parser.add_argument('--use_fsdp', action='store_true', default=False, help='Use Fully Sharded Data Parallel (FSDP)')
    parser.add_argument('--compile_model', action='store_true', default=False, help='Use torch.compile')

    args = parser.parse_args()

    # Set global flags
    global DEBUG, NO_TRACKER, NO_RESUME, NO_CUDA, SAVE_PROFILER_RESULTS, STAT_GRAD_NORM, TRACKER, USE_FSDP, COMPILE_MODEL
    DEBUG = args.debug
    NO_TRACKER = args.no_track_experiment
    NO_RESUME = args.no_resume
    NO_CUDA = args.no_cuda
    SAVE_PROFILER_RESULTS = args.trace_profiler
    STAT_GRAD_NORM = args.stat_grad_norm
    TRACKER = args.tracker
    USE_FSDP = args.use_fsdp
    COMPILE_MODEL = args.compile_model
    
    # Load hyperparameters from config file
    with open(args.config_file) as f:
        hyper_params = json.load(f)
    hyper_params['stage'] = args.experiment_stage

    # Determine run directory
    run_dir = args.run_dir if args.run_dir is not None else os.path.join(os.getcwd(), "runs")
    
    # Determine experiment name
    if args.experiment_name is None:
        # Use config file name as experiment name
        config_file_name = os.path.basename(args.config_file).split('.')[0]
        if args.experiment_stage is not None and args.experiment_stage in config_file_name:
            experiment_name = config_file_name.replace(f"-{args.experiment_stage}", "")
        else:
            experiment_name = config_file_name
    else:
        experiment_name = args.experiment_name

    # Determine experiment group
    if args.experiment_group is None:
        # Use the beginning of experiment name as experiment group
        experiment_group = experiment_name.split('-')[0]
    else:
        experiment_group = args.experiment_group
      
    # Prepare experiment arguments
    exp_args = {
        "project_name": args.project_name,
        "run_dir": run_dir,
        "experiment_name": experiment_name,
        "experiment_group": experiment_group,
        "experiment_stage": args.experiment_stage,
        "hyper_params": hyper_params,
    }

    return exp_args

def initialize_exp(
        project_name: str,
        run_dir: str,
        experiment_name: str,
        experiment_group: str,
        experiment_stage: str,
        hyper_params: Dict,
):
    # Use global flags directly
    global DEBUG, NO_TRACKER, NO_RESUME, NO_CUDA, SAVE_PROFILER_RESULTS, TRACKER, USE_FSDP

    if DEBUG:
        import datetime
        import time
        today = datetime.date.today()
        if NO_RESUME:
            now = time.strftime("%H-%M")
        else:
            now = time.strftime("%H")
        experiment_name = f"{experiment_name}-{today}-{now}"

        # modify hyper params for debug
        hyper_params['training']['num_steps'] = 1000
        hyper_params['training']['log_interval'] = 5
        # every save still rotates; only steps 500 and 1000 are kept (large models: ~0.3 GB per checkpoint)
        hyper_params['training']['chkpt_interval'] = 100
        # hyper_params['dataloader']['num_workers'] = 0
        hyper_params['dataloader']['batch_size'] = 10
        hyper_params['dataloader']['num_validation'] = 10
        # modify experiment name and group
        experiment_group = "debug"

    exp_dir = os.path.join(run_dir, experiment_group, experiment_name)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')

    # init profiler
    def trace_handler(prof:torch.profiler.profile):
        output = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
        print(output)
        if SAVE_PROFILER_RESULTS:
            prof.export_chrome_trace(os.path.join(exp_dir, f"trace-{accelerator.process_index}.json"))
    profile_kwargs = accelerate.ProfileKwargs(
        activities=["cpu", "cuda"],
        schedule_option={
            "wait": 1,
            "warmup": 1,
            "active": 1,
            "repeat": 1,
            "skip_first": 1,
        },
        on_trace_ready=trace_handler,
        with_flops=True,
        # record_shapes=True,
        # profile_memory=True,
    )

    # FSDP plugin configuration
    if USE_FSDP:
        fsdp_plugin = accelerate.FullyShardedDataParallelPlugin(
            cpu_offload=True,
            sharding_strategy="SHARD_GRAD_OP",  # or "FULL_SHARD"
            auto_wrap_policy="no_wrap",  # or "TRANSFORMER_BASED_WRAP"
            backward_prefetch="BACKWARD_PRE",
            use_orig_params=True,
            state_dict_type="FULL_STATE_DICT",  # or "SHARDED_STATE_DICT"
        )
    else:
        fsdp_plugin = None

    # init accelerator
    accelerator = Accelerator(
        log_with=None if NO_TRACKER else TRACKER,
        project_dir=exp_dir,
        cpu=NO_CUDA,
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,  # we use dataloader step as the global step
        gradient_accumulation_steps=hyper_params['training']['ga_steps'],
        kwargs_handlers=[profile_kwargs],
        fsdp_plugin=fsdp_plugin,
    )
    if accelerator.is_main_process:
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)
        # write the hyper params to the experiment dir
        with open(os.path.join(exp_dir, f'{experiment_stage}.json'), 'w') as f:
            json.dump(hyper_params, f, indent=4)

    # modify hyper_params to fit ga_steps and multiple gpus
    if hyper_params['dataloader']['batch_size'] % accelerator.num_processes != 0:
        accelerator.print(f"Warning: batch size={hyper_params['dataloader']['batch_size']} is not divisible by the number of gpus={accelerator.num_processes}, and is reduced to be divisible")
        hyper_params['dataloader']['batch_size'] = hyper_params['dataloader']['batch_size'] - hyper_params['dataloader']['batch_size'] % accelerator.num_processes
    if hyper_params['dataloader'].get('num_validation',0) % accelerator.num_processes != 0:
        accelerator.print(f"Warning: number of validation data={hyper_params['dataloader']['num_validation']} is not divisible by the number of gpus, and is reduced to be divisible")
        hyper_params['dataloader']['num_validation'] = hyper_params['dataloader']['num_validation'] - hyper_params['dataloader']['num_validation'] % accelerator.num_processes
    assert hyper_params['dataloader']['batch_size'] > accelerator.num_processes and hyper_params['dataloader']['num_validation'] > 0

    # log interval for fixed step of gradient optimize
    hyper_params['training']['log_interval'] *= hyper_params['training']['ga_steps']
    # chkpt interval for fixed step of logs
    hyper_params['training']['chkpt_interval'] *= hyper_params['training']['log_interval']
    # init checkpointor
    checkpointor = CheckPointor(checkpoint_dir, hyper_params['training']['chkpt_interval'], accelerator, prefix=experiment_stage)

    # initialize tracker
    accelerator.init_trackers(
        project_name,
        config=hyper_params if TRACKER=="wandb" else None,
        init_kwargs={
            "wandb": {
                'mode': 'online',
                'group': experiment_group,
                'name': experiment_name,
                'resume': checkpointor.has_checkpoint,
                'id': checkpointor.uuid,
                'dir': exp_dir,
            },
            "tensorboard": {
                "flush_secs": 60,
            },
        }
    )

    # show the process info
    local_hostname = socket.gethostname()
    if accelerator.is_local_main_process:
        print(f"launched local main process rank {accelerator.process_index} on device {accelerator.device} of host {local_hostname}")
    else:
        print(f"launched process with local rank {accelerator.local_process_index} on device {accelerator.device} of host {local_hostname}")

    # batch size for each gpu
    hyper_params['dataloader']['batch_size'] = hyper_params['dataloader']['batch_size'] // accelerator.num_processes

    # show the hyper parameters
    accelerator.print(f"Using {accelerator.num_processes} gpus")
    accelerator.print(f"Using {hyper_params['training']['ga_steps']} gradient accumulation steps")
    accelerator.print(f"Using {hyper_params['dataloader']['batch_size']} batch size for each gpu")
    accelerator.print(f"Using {hyper_params['training']['log_interval']} log interval")
    accelerator.print(f"Using {hyper_params['training']['chkpt_interval']} chkpt interval")

    return accelerator, checkpointor


if __name__ == "__main__":
    exp_args = parse_args()
    accelerator, checkpointor = initialize_exp(**exp_args)
    main(accelerator, checkpointor, exp_args["hyper_params"])
