import os
import shutil
import uuid
from typing import Dict, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.data_loader import skip_first_batches
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from torch.nn import Module
from torch.utils.data import DataLoader

from graphqec.benchmark.utils import fit_log_lfr

__all__ = ['QECDecoderTrainer', 'CheckPointor', 'EpochStepIterator', 'CurriculumTeacher']

class QECDecoderTrainer:
    def __init__(self,
                 model: Module,
                 optimizer: AcceleratedOptimizer,
                 scheduler: AcceleratedScheduler,
                 accelerator:Accelerator,
                 *,
                 loss_fn = 'bce',
                 clip_grad_norm = 1.0,
                 mask_abnormal = 5,
                 log_interval = 100,
                 stat_grad_norm = False,
                #  cycle_weight_bias = None,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator

        self.log_interval = log_interval
        self.stat_grad_norm = stat_grad_norm
        self.clip_grad_norm = clip_grad_norm
        self.mask_abnormal = mask_abnormal

        self.last_loss = None
        self.last_val_loss = None

        if loss_fn == 'bce':
            self._loss_fn = self._bce_loss
        elif loss_fn == 'focal':
            self._loss_fn = self._focal_loss
        else:
            raise NotImplementedError
        
    def _bce_loss(self,predictions:torch.Tensor,obs_flips:torch.Tensor) -> torch.Tensor:
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, obs_flips, reduction='none')
        return bce_loss
    
    def _focal_loss(self,predictions:torch.Tensor,obs_flips:torch.Tensor) -> torch.Tensor:
        # focal loss
        _gamma = 2
        _alpha = 0.25
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, obs_flips, reduction='none')
        p = torch.sigmoid(predictions)
        p_t = p*obs_flips + (1-p)*(1-obs_flips)
        focal_weight = (1 - p_t).pow(_gamma)
        alpha_weight = _alpha * obs_flips + (1-_alpha)*(1-obs_flips)
        loss = focal_weight * alpha_weight * bce_loss

        return loss

    def train_batch(self, syndromes: torch.Tensor, obs_flips: torch.Tensor, *, current_step: int):
        self.model.train()
        
        # [batch,cycle,obs]
        predictions: torch.Tensor = self.model(syndromes)
        # [batch,cycle]
        cycle_loss = self._loss_fn(predictions, obs_flips).mean(dim=-1) # the obs dim
        
        # mask out abnormal samples
        if self.mask_abnormal is not None:
            abnormal = cycle_loss > (cycle_loss.mean() + self.mask_abnormal*cycle_loss.std())
            cycle_loss[abnormal] = 0

        loss = cycle_loss.mean()
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients and self.clip_grad_norm is not None:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        if current_step % self.log_interval == 0 and self.accelerator.gradient_state.sync_gradients:
            with torch.no_grad():
                train_loss = self.accelerator.reduce(cycle_loss.detach(), reduction='mean').mean(dim=0)
                self.last_loss = train_loss.mean().item()

                self.accelerator.log({
                    "loss": self.last_loss,
                    'learning_rate': self.scheduler.get_last_lr()[0]}, step=current_step)

                if train_loss.numel() > 1:
                    # record loss for each cycle
                    for cycle,loss in enumerate(train_loss.tolist()):
                        self.accelerator.log({
                            f"loss/train_{cycle:02d}": loss,
                        },step=current_step)

                if self.stat_grad_norm:
                    for name, child_module in self.model.named_children():
                        grads = [param.grad.view(-1) for param in child_module.parameters() if param.grad is not None]
                        if grads:
                            grad_vector = torch.cat(grads)
                            grad_norm = torch.norm(grad_vector, p=2).item()
                        else:
                            grad_norm = 0.0
                
                        self.accelerator.log({
                            f"grad_norm/{name}": grad_norm,
                            }, step=current_step)

        self.optimizer.zero_grad()
        self.scheduler.step()

    @torch.inference_mode
    def benchmark_model(self, dataloader, step: int, *, cycle_step: int = 1):
        self.model.eval()
        loss = []
        err = []
        _cycle_loss = {}
        _cycle_err = {}
        for syndromes, obs_flips in dataloader:
            predictions = self.model(syndromes)
            raw_loss = self._loss_fn(predictions, obs_flips).mean(dim=-1) # [batch,cycle]
            predictions = torch.sigmoid(predictions).round()
            batch_err = (predictions != obs_flips)
            if len(obs_flips.shape) == 3:
                # label has cycle dim
                num_batch, num_cycle, num_obs = obs_flips.shape
                # obs_flips: [batch,cycle,obs]
                loss.append(raw_loss) # [batch,cycle]
                err.append(batch_err)   # [batch,cycle,obs]
            else:
                assert len(obs_flips.shape) == 2
                # syndomres: (encoding,cycle,readout,)
                cycle_syndrome = syndromes[1]
                num_batch, num_cycle, _ = cycle_syndrome.shape
                num_cycle += 1 # include the encoding cycle

                if num_cycle not in _cycle_loss:
                    _cycle_loss[num_cycle] = []
                    _cycle_err[num_cycle] = []
                
                _cycle_loss[num_cycle].append(raw_loss) # [batch,]
                _cycle_err[num_cycle].append(batch_err) # [batch,obs]

        if len(loss) == 0:
            # non-incremental benchmarking
            test_cycle = []
            for i in sorted(list(_cycle_loss.keys())):
                test_cycle.append(i)
                loss.append(torch.cat(_cycle_loss[i],dim=0))
                err.append(torch.cat(_cycle_err[i],dim=0))

            loss = torch.stack(loss,dim=1).mean(dim=0) # [cycle]
            err = torch.stack(err,dim=1) # [batch,cycle,obs]

        else:
            test_cycle = range(1,num_cycle*cycle_step+1, cycle_step)
            loss = torch.cat(loss,dim=0).mean(dim=0)
            err = torch.cat(err,dim=0)
        
        regid_err = err.any(dim=-1).mean(dim=0,dtype=torch.float32)
        err = err.mean(dim=0,dtype=torch.float32)

        test_cycle = torch.tensor(test_cycle).to(torch.long).numpy()
        loss = self.accelerator.reduce(loss, reduction='mean')
        err = self.accelerator.reduce(err, reduction='mean')
        regid_err = self.accelerator.reduce(regid_err, reduction='mean')

        # record the latest cycle acc for each logical qubit
        for qubit_idx,e in enumerate(err[-1,:]):
            self.accelerator.log({
                f'qubit_error/q_{qubit_idx:02d}': e
            }, step=step)

        err = err.mean(dim=-1)

        # record acc for each cycle
        for c, (l,e,r) in enumerate(zip(loss.tolist(), err.tolist(), regid_err.tolist())):
            self.accelerator.log({
                f'loss/val_{c:02d}': l,
                f'error_rate/val_{c:02d}': e,
                f'regid_error_rate/val_{c:02d}': r
            }, step=step)

        # fit per-round error rate
        fids = (1 - 2 * err).squeeze().detach().cpu().numpy()
        f0,lfr,lfrstd = fit_log_lfr(fids[1:],rounds=test_cycle[1:])
        val_loss = loss.mean().item()
        self.last_val_loss = val_loss
        self.accelerator.log({
            'val_loss': val_loss,
            "lfr":lfr
        }, step=step)

        return f0,lfr,lfrstd
        # return 1 - (1-r) ** (1/(c+1))


class CheckPointor:
    def __init__(self, checkpoint_dir: str, checkpoint_interval: int, accelerator: Accelerator, prefix: str = "chkpt"):
        # absolute, so that the *_latest symlinks resolve for a relative --run_dir
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.accelerator = accelerator
        self.prefix = prefix

        # Define checkpoint-related paths with the prefix
        self.latest = {
            "chkpt_latest": os.path.join(checkpoint_dir, f"{prefix}_latest"),
            "lock_latest": os.path.join(checkpoint_dir, f"{prefix}_lock_latest"),
            "chkpt_latest_bac": os.path.join(checkpoint_dir, f"{prefix}_latest_bac"),
            "chkpt_latest_bac_bac": os.path.join(checkpoint_dir, f"{prefix}_latest_bac_bac"),
        }

        # Ensure checkpoint directory exists and initialize UUID
        if self.accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            if not self.has_uuid:
                self._write_uuid()

        accelerator.wait_for_everyone()
        self.uuid = self._read_uuid()
        self.checkpoints, self.special_checkpoints = self._find_checkpoints()

        # Define helper functions for file operations
        self._rename = self._create_file_op("rename")
        self._rm = self._create_file_op("remove")
        self._symlink = self._create_file_op("symlink")
        self._cp = self._create_file_op("copy")

    @property
    def has_checkpoint(self) -> bool:
        """Check if there are any checkpoints available."""
        return bool(self.checkpoints or self.special_checkpoints)

    def save(self, global_step: int | str):
        """
        Save the current state as a checkpoint.
        If global_step is an integer, it saves a regular checkpoint.
        If global_step is a string, it saves a special checkpoint and overwrites if it already exists.
        """
        if isinstance(global_step, int):
            # Regular checkpoint saving logic
            self._safe_checkpointing(global_step)
        elif isinstance(global_step, str):
            # Special checkpoint saving logic
            special_chkpt_path = os.path.join(self.checkpoint_dir, f"{self.prefix}_{global_step}")
            
            # Remove existing special checkpoint if it exists
            if os.path.exists(special_chkpt_path):
                self._rm(special_chkpt_path)
            self.accelerator.wait_for_everyone()
            
            # Save the current state as a special checkpoint
            self.accelerator.save_state(special_chkpt_path)
            
            # Update the special checkpoints dictionary
            self.special_checkpoints[global_step] = special_chkpt_path
        else:
            raise ValueError("global_step must be either an integer or a string.")

        # Synchronize all processes and update checkpoint lists
        self.accelerator.wait_for_everyone()
        self.checkpoints, self.special_checkpoints = self._find_checkpoints()

    def load(self, idx: Optional[int] = None):
        """Load a checkpoint by index or special name."""
        target = self._resolve_load_target(idx)
        self.accelerator.load_state(target)

    @property
    def latest_step(self) -> int:
        """Get the step number of the latest checkpoint."""
        if not self.has_checkpoint:
            return -1
        latest_path = self.special_checkpoints.get("latest")
        _, step = os.path.basename(latest_path).split("_", 1)
        return int(step)

    @property
    def has_uuid(self) -> bool:
        """Check if the UUID file exists."""
        return os.path.exists(os.path.join(self.checkpoint_dir, f"uuid-{self.prefix}"))

    def _write_uuid(self):
        """Write a unique UUID to the checkpoint directory."""
        with open(os.path.join(self.checkpoint_dir, f"uuid-{self.prefix}"), "w") as f:
            f.write(uuid.uuid1().hex)

    def _read_uuid(self) -> str:
        """Read the UUID from the checkpoint directory."""
        with open(os.path.join(self.checkpoint_dir, f"uuid-{self.prefix}"), "r") as f:
            return f.readline().strip()

    def _find_checkpoints(self) -> Tuple[Dict[int, str], Dict[str, str]]:
        """Find all checkpoints and special checkpoints in the directory."""
        checkpoints = {}
        special_checkpoints = {}
        for entry in os.listdir(self.checkpoint_dir):
            path = os.path.join(self.checkpoint_dir, entry)
            if entry.startswith(f"{self.prefix}_"):
                try:
                    kind, idx = entry.split("_", 1)
                    if idx.isdigit():
                        checkpoints[int(idx)] = path
                    else:
                        special_checkpoints[idx] = os.readlink(path) if os.path.islink(path) else path
                except ValueError as e:
                    self.accelerator.print(f"Invalid checkpoint entry: {entry}, error: {e}")
                except OSError as e:
                    self.accelerator.print(f"Failed to process checkpoint: {entry}, error: {e}")
        return checkpoints, special_checkpoints

    def _create_file_op(self, operation: str):
        """Create a file operation function with safety checks."""
        def wrapper(src, dst=None):
            if src == self.checkpoint_dir or not src.startswith(self.checkpoint_dir):
                raise ValueError(f"Invalid source path: {src}")
            if dst and (dst == self.checkpoint_dir or not dst.startswith(self.checkpoint_dir)):
                raise ValueError(f"Invalid destination path: {dst}")

            if operation == "rename":
                os.rename(src, dst)
            elif operation == "remove":
                if not os.path.isdir(src):  # Check for empty directory
                    os.remove(src)
                else:
                    shutil.rmtree(src)
            elif operation == "symlink":
                os.symlink(src, dst)
            elif operation == "copy":
                shutil.copytree(src, dst)

        return self.accelerator.on_main_process(wrapper)

    def _safe_checkpointing(self, global_step: int):
        """Perform safe checkpointing with backup mechanisms."""
        current_chkpt = os.path.join(self.checkpoint_dir, f"{self.prefix}_{global_step}")
        lock_path = self.latest["lock_latest"]

        # Handle interrupted checkpointing
        if os.path.exists(self.latest["chkpt_latest"]):
            self._cp(self.latest["chkpt_latest"], lock_path)
        else:
            if self.has_checkpoint:
                raise RuntimeError("Checkpoint directory is not empty at the start of training.")
            if self.accelerator.is_main_process:
                with open(lock_path, "w") as f:
                    f.write("in middle of checkpointing")

        # Save the current state
        self.accelerator.save_state(current_chkpt)

        # Manage backup checkpoints
        if os.path.exists(self.latest["chkpt_latest_bac_bac"]):
            self._handle_backup_cleanup()
        if os.path.exists(self.latest["chkpt_latest_bac"]):
            self._rename(self.latest["chkpt_latest_bac"], self.latest["chkpt_latest_bac_bac"])
        if os.path.exists(self.latest["chkpt_latest"]):
            self._rename(self.latest["chkpt_latest"], self.latest["chkpt_latest_bac"])

        # Update symlinks and clean up lock
        self._symlink(current_chkpt, self.latest["chkpt_latest"])
        self._rm(lock_path)

    def _handle_backup_cleanup(self):
        latest_bac_bac = self.special_checkpoints.get("latest_bac_bac")
        if latest_bac_bac:
            try:
                chkpt_idx = int(os.path.basename(latest_bac_bac).split("_")[-1])
                if chkpt_idx % self.checkpoint_interval != 0:
                    self._rm(latest_bac_bac)
            except (ValueError, IndexError):
                self.accelerator.print("Invalid format for latest_bac_bac")

    def _resolve_load_target(self, idx: Optional[int]) -> str:
        """Resolve the target checkpoint path for loading."""
        if idx is None:
            if os.path.exists(self.latest["lock_latest"]):
                self._recover_interrupted_checkpoint()
            return self.latest["chkpt_latest"]
        elif isinstance(idx, int):
            return self.checkpoints[idx]
        else:
            return self.special_checkpoints[idx]

    def _recover_interrupted_checkpoint(self):
        try:
            if os.path.exists(self.latest["chkpt_latest"]):
                self._rm(self.latest["chkpt_latest"])
            if os.path.exists(self.latest["lock_latest"]):
                self._rename(self.latest["lock_latest"], self.latest["chkpt_latest"])
            if os.path.exists(self.latest["chkpt_latest_bac_bac"]):
                self._rename(self.latest["chkpt_latest_bac_bac"], self.latest["chkpt_latest_bac"])
            self.special_checkpoints["latest"] = self.special_checkpoints.get("latest_bac", "")
            self.special_checkpoints["latest_bac"] = self.special_checkpoints.get("latest_bac_bac", "")
            self.special_checkpoints.pop("latest_bac_bac", None)
        except Exception as e:
            self.accelerator.print(f"Failed to recover interrupted checkpoint: {e}")

class CurriculumTeacher:
    """
    sample batches from multiple dataloaders while acting like a single one,
    and the choosen dataset should be the same across distributed workers
    TODO: biased sampling as curriculum learning
    """
    def __init__(self, dataloaders: DataLoader, seed: int = 42) -> None:
        super().__init__()

        assert len(dataloaders)>1, "only one dataloader is provided"
        self.num_samples = sum([len(dataloader) for dataloader in dataloaders])
        self.dataloaders = dataloaders

        self.curriculum_seed = seed
        self.curriculum_generator = torch.Generator('cpu')
        self.curriculum_generator.manual_seed(self.curriculum_seed)

        self.counts = None
        self._iterators = None
        self._skipped_dataloaders = None

    def __len__(self):
        return self.num_samples
    
    def __iter__(self):
        if self.counts is None or sum(self.counts) == 0:
            self.counts = torch.tensor([len(dataloader) for dataloader in self.dataloaders],dtype=torch.float)
            self._iterators = [iter(d) for d in self.dataloaders]
            self._skipped_dataloaders = None
        elif self._iterators is None:
            assert self._skipped_dataloaders is not None
            # load from checkpoint, the iterators are not saved
            # NOTE the iterators are reset here, meaning the dataset will be resampled from the beginning
            self._iterators = [iter(d) for d in self._skipped_dataloaders]
        for _ in range(torch.sum(self.counts).to(torch.long).item()):
            # choose a dataset 
            assert sum(self.counts) > 0
            idx = torch.multinomial(self.counts, 1, generator=self.curriculum_generator).item()
            self.counts[idx] -= 1
            yield next(self._iterators[idx])

    def state_dict(self):
        ignores = ['curriculum_generator','dataloaders','_iterators','_skipped_dataloaders']
        return {k:v for k,v in self.__dict__.items() if k not in ignores}
    
    def load_state_dict(self, state_dict):
        assert state_dict['num_samples'] == self.num_samples
        self.__dict__.update(state_dict)
        self.curriculum_generator.manual_seed(self.curriculum_seed)
        if self.counts is not None and sum(self.counts) != 0:
            self._skipped_dataloaders = [
                skip_first_batches(dataloader, len(dataloader)-self.counts[i]) 
                for i,dataloader in enumerate(self.dataloaders)
                ]
            
class EpochStepIterator:
    """iterate step over epoch"""
    def __init__(self,dataloader: DataLoader | CurriculumTeacher, num_total_steps, num_total_epochs, verbose:bool = False) -> None:
        self.dataloader = dataloader
        self.num_total_steps = num_total_steps
        self.num_total_epochs = num_total_epochs
        self.verbose = verbose

        self.current_step = 0
        self.current_epoch = 0

        self._skipped_dataloader = None
        self._iterator = None

    def __iter__(self):
        while self.current_step < self.num_total_steps and self.current_epoch < self.num_total_epochs:
            if self.verbose:
                print(f'epoch {self.current_epoch} begins at step {self.current_step}')
            if self._skipped_dataloader is not None:
                self._iterator = iter(self._skipped_dataloader)
                self._skipped_dataloader = None
            else:
                self._iterator = iter(self.dataloader)

            for batch in self._iterator:
                self.current_step += 1
                yield self.current_step, self.current_epoch, batch
                if self.current_step >= self.num_total_steps:
                    break
            if self.verbose:
                print(f'epoch {self.current_epoch} ends at step {self.current_step}')
            self.current_epoch += 1
        if self.verbose:
            print(f'iteration ends at step {self.current_step}, epoch {self.current_epoch}')

    def __len__(self):
        return self.num_total_steps
    
    def state_dict(self):
        return {"current_step": self.current_step, "current_epoch":self.current_epoch}
    
    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
        self.current_epoch = state_dict["current_epoch"]
        if isinstance(self.dataloader,DataLoader):
            skip_first_batches(self.dataloader, self.current_step)