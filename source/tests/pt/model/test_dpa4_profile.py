# SPDX-License-Identifier: LGPL-3.0-or-later
"""Opt-in end-to-end profiler for the water DPA4 energy/force path.

Run manually with::

    DP_RUN_DPA4_PROFILE=1 DP_DPA4_PROFILE_DIR=/tmp/dpa4-profile \
        pytest -s source/tests/pt/model/test_dpa4_profile.py

The trace includes test-local ``record_function`` ranges for the descriptor,
edge cache, environment FiLM, GIE, interaction blocks/SO(2), readout and
fitting, alongside ordinary ATen and device-kernel events. No production
instrumentation is installed.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

_RUN_PROFILE = os.environ.get("DP_RUN_DPA4_PROFILE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _read_water_lammps_data(
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read coordinates, types and orthorhombic box from the example data file."""
    bounds: dict[str, tuple[float, float]] = {}
    atoms: list[tuple[int, int, float, float, float]] = []
    in_atoms = False

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for axis in ("x", "y", "z"):
            if stripped.endswith(f"{axis}lo {axis}hi"):
                fields = stripped.split()
                bounds[axis] = (float(fields[0]), float(fields[1]))
                break
        if stripped.startswith("Atoms"):
            in_atoms = True
            continue
        if not in_atoms:
            continue
        fields = stripped.split()
        if len(fields) >= 5 and fields[0].isdigit() and fields[1].isdigit():
            atoms.append(
                (
                    int(fields[0]),
                    int(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                    float(fields[4]),
                )
            )
        elif atoms:
            break

    if len(bounds) != 3 or not atoms:
        raise ValueError(f"could not parse LAMMPS water geometry from {path}")
    atoms.sort(key=lambda row: row[0])
    coordinates = torch.tensor(
        [[row[2], row[3], row[4]] for row in atoms],
        dtype=torch.float32,
        device="cpu",
    ).unsqueeze(0)
    atom_types = torch.tensor(
        [[row[1] - 1 for row in atoms]], dtype=torch.int64, device="cpu"
    )
    lengths = [bounds[axis][1] - bounds[axis][0] for axis in ("x", "y", "z")]
    box = torch.diag(torch.tensor(lengths, dtype=torch.float32, device="cpu")).reshape(
        1, 9
    )
    return coordinates, atom_types, box


@contextlib.contextmanager
def _record_method(owner: Any, method_name: str, label: str):
    """Temporarily wrap one bound method in a profiler range."""
    original = getattr(owner, method_name, None)
    if not callable(original):
        yield
        return

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)

    setattr(owner, method_name, wrapped)
    try:
        yield
    finally:
        setattr(owner, method_name, original)


@contextlib.contextmanager
def _record_function_attribute(module: Any, attribute: str, label: str):
    """Temporarily annotate a module-level function call with a profiler range."""
    original = getattr(module, attribute)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)

    setattr(module, attribute, wrapped)
    try:
        yield
    finally:
        setattr(module, attribute, original)


@contextlib.contextmanager
def _dpa4_profile_ranges(model: torch.nn.Module):
    """Add test-only ranges around major eager DPA4 stages."""
    atomic_model = getattr(model, "atomic_model", None)
    descriptor = getattr(atomic_model, "descriptor", None)
    if descriptor is None:
        raise RuntimeError("the loaded model does not expose a DPA4 descriptor")

    dpa4_impl = importlib.import_module("deepmd.dpmodel.descriptor.dpa4")
    make_model_impl = importlib.import_module("deepmd.pt_expt.model.make_model")
    transform_output_impl = importlib.import_module(
        "deepmd.pt_expt.model.transform_output"
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            _record_function_attribute(
                make_model_impl,
                "build_neighbor_graph_for_method",
                "dpa4/neighbor_graph",
            )
        )
        stack.enter_context(
            _record_function_attribute(
                transform_output_impl, "task_deriv_one", "dpa4/force_autograd"
            )
        )
        stack.enter_context(
            _record_function_attribute(
                dpa4_impl, "_edge_cache_from_arrays", "dpa4/edge_cache"
            )
        )
        stack.enter_context(_record_method(model, "forward", "dpa4/model.forward"))
        stack.enter_context(
            _record_method(descriptor, "_run_graph", "dpa4/descriptor.graph")
        )
        stack.enter_context(
            _record_method(descriptor, "_forward_blocks", "dpa4/interaction_blocks")
        )
        stack.enter_context(
            _record_method(descriptor, "_apply_readout", "dpa4/readout")
        )
        for module, method, label in (
            (descriptor.type_embedding, "forward", "dpa4/type_embedding"),
            (descriptor.edge_envelope, "forward", "dpa4/cutoff_envelope"),
            (descriptor.radial_basis, "forward", "dpa4/radial_basis"),
            (descriptor.wigner_calc, "call", "dpa4/wigner"),
            (descriptor.radial_embedding, "forward", "dpa4/radial_mlp"),
        ):
            stack.enter_context(_record_method(module, method, label))

        env_seed = getattr(descriptor, "env_seed_embedding", None)
        if env_seed is not None:
            stack.enter_context(
                _record_method(env_seed, "call", "dpa4/environment_film")
            )
        gie = getattr(descriptor, "gie", None)
        if gie is not None:
            stack.enter_context(_record_method(gie, "call", "dpa4/gie"))

        for index, block in enumerate(descriptor.blocks):
            stack.enter_context(_record_method(block, "forward", f"dpa4/block_{index}"))
            conv = getattr(block, "so2_conv", None)
            if conv is None:
                continue
            stack.enter_context(_record_method(conv, "call", f"dpa4/block_{index}/so2"))
            for method_name, stage in (
                ("_rotate_mix", "rotate_mix"),
                ("_rotate_back", "rotate_back"),
                ("attention_weights", "attention_weights"),
                ("_attention_weighted_reduce", "attention_reduce"),
            ):
                stack.enter_context(
                    _record_method(
                        conv,
                        method_name,
                        f"dpa4/block_{index}/so2/{stage}",
                    )
                )

        fitting = getattr(atomic_model, "fitting_net", None)
        if fitting is not None:
            stack.enter_context(_record_method(fitting, "forward", "dpa4/fitting"))
        yield


def _event_device_us(event: Any) -> float:
    """Read device-time fields across PyTorch profiler API versions."""
    for field in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, field, None)
        if value is not None:
            return float(value)
    return 0.0


def _event_row(event: Any, steps: int) -> dict[str, Any]:
    return {
        "name": event.key,
        "calls": int(event.count),
        "self_cpu_us": float(event.self_cpu_time_total),
        "total_cpu_us": float(event.cpu_time_total),
        "self_device_us": _event_device_us(event),
        "self_cpu_us_per_step": float(event.self_cpu_time_total) / steps,
        "self_device_us_per_step": _event_device_us(event) / steps,
        "self_cpu_bytes": int(getattr(event, "self_cpu_memory_usage", 0)),
    }


def _load_profile_model(config_path: Path, checkpoint_path: Path, device):
    """Construct the pt_expt model and restore the example trainer checkpoint."""
    from deepmd.pt_expt.model.get_model import get_model
    from deepmd.pt_expt.train.wrapper import ModelWrapper

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model", checkpoint)
    extra_state = state_dict.get("_extra_state", {})
    model_params = extra_state.get("model_params")
    if not isinstance(model_params, dict):
        model_params = json.loads(config_path.read_text())["model"]
    model_params = copy.deepcopy(model_params)
    # The pt test package sets an invalid default device to catch implicit
    # placement; construct the checkpoint model explicitly on CPU, then move it.
    with torch.device("cpu"):
        model = get_model(copy.deepcopy(model_params))
        wrapper = ModelWrapper(model, model_params=copy.deepcopy(model_params))
        wrapper.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_params


@pytest.mark.skipif(
    not _RUN_PROFILE,
    reason="set DP_RUN_DPA4_PROFILE=1 to run the opt-in DPA4 profiler",
)
def test_profile_water_dpa4_energy_force(tmp_path: Path) -> None:
    """Capture a real water DPA4 energy/force trace and rank bottlenecks."""
    repo_root = Path(__file__).resolve().parents[4]
    example_dir = repo_root / "examples" / "water" / "dpa4" / "lmp"
    config_path = Path(
        os.environ.get("DP_DPA4_PROFILE_CONFIG", example_dir / "input.json")
    ).expanduser()
    checkpoint_path = Path(
        os.environ.get("DP_DPA4_PROFILE_CHECKPOINT", example_dir / "pretrained.pt")
    ).expanduser()
    geometry_path = Path(
        os.environ.get("DP_DPA4_PROFILE_GEOMETRY", example_dir / "water.lmp")
    ).expanduser()
    output_dir = Path(
        os.environ.get("DP_DPA4_PROFILE_DIR", tmp_path / "dpa4_profile")
    ).expanduser()
    if not checkpoint_path.is_file():
        pytest.fail(f"DPA4 profile checkpoint does not exist: {checkpoint_path}")
    if not config_path.is_file():
        pytest.fail(f"DPA4 profile config does not exist: {config_path}")
    if not geometry_path.is_file():
        pytest.fail(f"DPA4 profile geometry does not exist: {geometry_path}")

    device_name = os.environ.get("DP_DPA4_PROFILE_DEVICE")
    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, model_params = _load_profile_model(config_path, checkpoint_path, device)
    coordinates, atom_types, box = _read_water_lammps_data(geometry_path)
    atom_limit = int(os.environ.get("DP_DPA4_PROFILE_ATOMS", "0"))
    if atom_limit > 0 and atom_limit < coordinates.shape[1]:
        generator = torch.Generator().manual_seed(20261102)
        selection = torch.randperm(
            coordinates.shape[1], generator=generator, device="cpu"
        )[:atom_limit]
        coordinates = coordinates[:, selection]
        atom_types = atom_types[:, selection]

    descriptor_precision = str(model_params["descriptor"].get("precision", "float32"))
    input_dtype = torch.float64 if descriptor_precision == "float64" else torch.float32
    coordinates = coordinates.to(device=device, dtype=input_dtype)
    atom_types = atom_types.to(device=device)
    box = box.to(device=device, dtype=input_dtype)

    warmup_steps = max(0, int(os.environ.get("DP_DPA4_PROFILE_WARMUP", "2")))
    profile_steps = max(1, int(os.environ.get("DP_DPA4_PROFILE_STEPS", "5")))

    def run_inference():
        with torch.device(device):
            coord = coordinates.detach().clone().requires_grad_(True)
            return model(coord, atom_types, box)

    for _ in range(warmup_steps):
        outputs = run_inference()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "dpa4_trace.json"
    summary_path = output_dir / "dpa4_profile_summary.json"
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        with _dpa4_profile_ranges(model):
            for _ in range(profile_steps):
                with torch.profiler.record_function("dpa4/inference.energy_force"):
                    outputs = run_inference()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                profiler.step()

    energy = outputs.get("energy")
    force = outputs.get("force")
    assert energy is not None and torch.isfinite(energy).all().item()
    assert force is not None and torch.isfinite(force).all().item()

    events = profiler.key_averages()
    rows = [_event_row(event, profile_steps) for event in events]
    stage_rows = [row for row in rows if row["name"].startswith("dpa4/")]
    operator_rows = [row for row in rows if not row["name"].startswith("dpa4/")]
    stage_sort_field = (
        "self_device_us"
        if any(row["self_device_us"] > 0.0 for row in stage_rows)
        else "self_cpu_us"
    )
    operator_sort_field = (
        "self_device_us"
        if any(row["self_device_us"] > 0.0 for row in operator_rows)
        else "self_cpu_us"
    )
    stage_rows.sort(key=lambda row: row[stage_sort_field], reverse=True)
    operator_rows.sort(key=lambda row: row[operator_sort_field], reverse=True)
    operator_rows = operator_rows[:30]

    profiler.export_chrome_trace(str(trace_path))
    report = {
        "model_config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "geometry": str(geometry_path),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "atom_count": int(coordinates.shape[1]),
        "dtype": str(coordinates.dtype),
        "warmup_steps": warmup_steps,
        "profile_steps": profile_steps,
        "tilelang_dpa4_enabled": os.environ.get("DP_TILELANG_DPA4", "0"),
        "trace_file": str(trace_path),
        "stage_sort_metric": stage_sort_field,
        "operator_sort_metric": operator_sort_field,
        "stage_hotspots": stage_rows,
        "operator_hotspots": operator_rows,
    }
    summary_path.write_text(json.dumps(report, indent=2) + "\n")

    sys.stdout.write(f"DPA4 profile trace: {trace_path}\n")
    sys.stdout.write(f"DPA4 profile summary: {summary_path}\n")
    sys.stdout.write(f"Top DPA4 stage ranges by {stage_sort_field}:\n")
    for row in stage_rows[:20]:
        sys.stdout.write(
            f"  {row['name']}: {row[stage_sort_field] / profile_steps:.3f} us/step "
            f"self, {row['calls']} calls\n"
        )
    sys.stdout.write(f"Top operators by {operator_sort_field}:\n")
    for row in operator_rows[:20]:
        sys.stdout.write(
            f"  {row['name']}: {row[operator_sort_field] / profile_steps:.3f} us/step "
            f"self, {row['calls']} calls\n"
        )
    sys.stdout.flush()

    stage_names = {row["name"] for row in stage_rows}
    assert "dpa4/inference.energy_force" in stage_names
    assert "dpa4/neighbor_graph" in stage_names
    assert "dpa4/edge_cache" in stage_names
    assert "dpa4/descriptor.graph" in stage_names
    assert "dpa4/force_autograd" in stage_names
    assert trace_path.stat().st_size > 0
    assert summary_path.stat().st_size > 0
