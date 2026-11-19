"""Training-path regressions: arguments the data loader passes and checkpoints under a relative run dir."""

import os

from graphqec.qecc import get_code


def test_bb_circuit_accepts_training_parity_argument():
    code = get_code("ETHBBCode", profile_name="[[72,12,6]]")
    plain = code.get_syndrome_circuit(2, physical_error_rate=0.001)
    assert code.get_syndrome_circuit(2, physical_error_rate=0.001, parity=None) == plain


def test_checkpoint_symlinks_resolve_for_relative_run_dir(tmp_path, monkeypatch):
    from accelerate import Accelerator
    from graphqec.decoder.nn.trainer import CheckPointor

    monkeypatch.chdir(tmp_path)
    accelerator = Accelerator(cpu=True)
    accelerator.register_for_checkpointing(type("State", (), {"state_dict": lambda self: {},
                                                                "load_state_dict": lambda self, s: None})())
    checkpointor = CheckPointor(os.path.join("runs", "checkpoints"), 1, accelerator, prefix="pretrain")
    for step in (1, 2, 3):
        checkpointor.save(step)
    latest = tmp_path / "runs" / "checkpoints" / "pretrain_latest"
    assert latest.is_symlink() and latest.resolve() == (tmp_path / "runs" / "checkpoints" / "pretrain_3")
