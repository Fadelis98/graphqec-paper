"""Strict key/shape compatibility between a released checkpoint and the public model definition."""

from __future__ import annotations

from pathlib import Path

ALLOWED_EXTRA_KEYS = frozenset({"data_to_check", "data_to_logical"})


def _strip(state):
    return {k.removeprefix("_orig_mod."): v for k, v in state.items()}


def checkpoint_report(model, checkpoint) -> dict:
    """Compare ``model.state_dict()`` with a ``model.safetensors`` file (or a directory holding one).

    Two legacy graph buffers (``data_to_check``, ``data_to_logical``) that some BB checkpoints carry are
    the only tolerated extra entries; every model tensor must be present with the identical shape.
    """
    from safetensors.torch import load_file

    path = Path(checkpoint)
    if path.is_dir():
        path = path / "model.safetensors"
    ckpt = _strip(load_file(str(path)))
    model_state = _strip(model.state_dict())
    missing = sorted(set(model_state) - set(ckpt))
    extra = sorted(set(ckpt) - set(model_state))
    disallowed = sorted(set(extra) - ALLOWED_EXTRA_KEYS)
    shape = sorted(k for k in set(model_state) & set(ckpt) if tuple(model_state[k].shape) != tuple(ckpt[k].shape))
    return {"checkpoint": str(path), "model_keys": len(model_state), "checkpoint_keys": len(ckpt),
            "missing": missing, "extra_allowed": sorted(set(extra) & ALLOWED_EXTRA_KEYS),
            "extra_disallowed": disallowed, "shape_mismatches": shape,
            "compatible": not missing and not disallowed and not shape}


def build_model(code_config: dict, model_config: dict):
    """Build the public model (no weights) for a code/model configuration pair."""
    from graphqec.decoder.nn import get_model
    from graphqec.qecc import get_code

    code_config = dict(code_config)
    code = get_code(code_config.pop("code_type"), **code_config)
    model_config = {k: v for k, v in model_config.items() if k not in ("chkpt",)}
    name = model_config.pop("name")
    return get_model(name=name, tanner_graph=code.get_tanner_graph(), **model_config)
