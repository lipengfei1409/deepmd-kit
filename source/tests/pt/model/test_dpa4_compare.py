# SPDX-License-Identifier: LGPL-3.0-or-later
"""Opt-in comparison of DeepMD-kit DPA4 and its TileLang inference path.

This compares the same pt_expt water checkpoint with TileLang dispatch disabled
and enabled. It measures numerical agreement and steady-state energy+force
latency; it does not establish scientific accuracy because the fixture has no
reference labels.

From the DeepMD-kit checkout, add myproject's Python package to PYTHONPATH and
run::

    DP_RUN_DPA4_COMPARE=1 pytest -s source/tests/pt/model/test_dpa4_compare.py

``DP_TILELANG_DPA4`` is managed by the test. Set ``DP_DPA4_COMPARE_DIR`` to
choose where the JSON comparison report is written.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import torch

_RUN_COMPARE = os.environ.get("DP_RUN_DPA4_COMPARE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_TILELANG_FLAGS = (
    "DP_TILELANG_DPA4",
    "DP_TILELANG_RADIAL_BASIS",
    "DP_TILELANG_C3_ENVELOPE",
    "DP_TILELANG_EDGE_GEOMETRY",
    "DP_TILELANG_EDGE_DEGREE",
    "DP_TILELANG_EDGE_QUATERNION",
    "DP_TILELANG_EDGE_TYPE_FEATURES",
    "DP_TILELANG_WIGNER_L1",
    "DP_TILELANG_ZONAL_SCATTER",
    "DP_TILELANG_ENV_SEED_MATRIX",
    "DP_TILELANG_ENV_SEED_SCATTER",
    "DP_TILELANG_ENV_SEED_GRAM",
    "DP_TILELANG_SO2_ROTATION_L1",
    "DP_TILELANG_SO2_RADIAL_SCALE",
    "DP_TILELANG_SO2_LINEAR_L1",
    "DP_TILELANG_ATTENTION_SOFTMAX",
    "DP_TILELANG_ATTENTION_AGGREGATE",
    "DP_TILELANG_GRID_PRODUCT",
)
_OTHER_ACCELERATION_FLAGS = (
    "DP_TRITON_INFER",
    "DP_CUDA_INFER",
    "DP_AMP_INFER",
    "DP_CUTE_INFER",
    "DP_CUTILE_INFER",
)


@contextlib.contextmanager
def _inference_backend(tilelang_enabled: bool):
    """Set selectors before model construction, then restore caller state."""
    names = _TILELANG_FLAGS + _OTHER_ACCELERATION_FLAGS
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "0"
        if tilelang_enabled:
            os.environ["DP_TILELANG_DPA4"] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _read_water_lammps_data(
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read the example's coordinates, types, and orthorhombic box."""
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
        [[row[2], row[3], row[4]] for row in atoms], dtype=torch.float32
    ).unsqueeze(0)
    atom_types = torch.tensor([[row[1] - 1 for row in atoms]], dtype=torch.int64)
    lengths = [bounds[axis][1] - bounds[axis][0] for axis in ("x", "y", "z")]
    box = torch.diag(torch.tensor(lengths, dtype=torch.float32)).reshape(1, 9)
    return coordinates, atom_types, box


def _require_comparison_runtime() -> None:
    """Skip only when the opt-in target runtime is known to be unavailable."""
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("DPA4 TileLang comparison requires a HIP/DCU PyTorch device")
    if importlib.util.find_spec("tilelang") is None:
        pytest.skip("DPA4 TileLang comparison requires TileLang to be installed")
    try:
        importlib.import_module("deepmd.lib")
    except ModuleNotFoundError as error:
        if error.name in {"deepmd", "deepmd.lib"}:
            pytest.skip(
                "compiled DeepMD-kit runtime deepmd.lib is unavailable: "
                f"{error}"
            )
        raise


def _load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    """Construct one pt_expt model and restore the same trainer checkpoint."""
    from deepmd.pt_expt.model.get_model import get_model
    from deepmd.pt_expt.train.wrapper import ModelWrapper

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model_params = state_dict.get("_extra_state", {}).get("model_params")
    if not isinstance(model_params, dict):
        model_params = json.loads(config_path.read_text())["model"]
    model_params = copy.deepcopy(model_params)
    # pt_expt tests deliberately reject implicit device placement.
    with torch.device("cpu"):
        model = get_model(copy.deepcopy(model_params))
        wrapper = ModelWrapper(model, model_params=copy.deepcopy(model_params))
        wrapper.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_params


def _run_energy_force(
    model: torch.nn.Module,
    coordinates: torch.Tensor,
    atom_types: torch.Tensor,
    box: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run the ordinary pt_expt energy-and-coordinate-force inference path."""
    with torch.device(device), torch.enable_grad():
        coord = coordinates.detach().clone().requires_grad_(True)
        return model(coord, atom_types, box)


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _measure_variant(
    model: torch.nn.Module,
    coordinates: torch.Tensor,
    atom_types: torch.Tensor,
    box: torch.Tensor,
    device: torch.device,
    warmup_steps: int,
    repeat_steps: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Separate first-call/JIT time from synchronized steady-state samples."""
    _synchronize(device)
    start = time.perf_counter()
    outputs = _run_energy_force(model, coordinates, atom_types, box, device)
    _synchronize(device)
    first_call_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(warmup_steps):
        outputs = _run_energy_force(model, coordinates, atom_types, box, device)
    _synchronize(device)

    samples_ms: list[float] = []
    for _ in range(repeat_steps):
        _synchronize(device)
        start = time.perf_counter()
        outputs = _run_energy_force(model, coordinates, atom_types, box, device)
        _synchronize(device)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return outputs, {
        "first_call_ms": first_call_ms,
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "p95_ms": _percentile(samples_ms, 0.95),
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _error_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference64 = reference.detach().to(device="cpu", dtype=torch.float64)
    actual64 = actual.detach().to(device="cpu", dtype=torch.float64)
    difference = actual64 - reference64
    epsilon = 1.0e-12
    return {
        "max_abs": float(difference.abs().max().item()),
        "mae": float(difference.abs().mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "relative_l2": float(
            difference.norm().item() / max(reference64.norm().item(), epsilon)
        ),
    }


def _assert_finite_outputs(variant: str, outputs: dict[str, torch.Tensor]) -> None:
    for field in ("energy", "force"):
        value = outputs.get(field)
        if value is None:
            raise AssertionError(f"{variant} inference did not return {field}")
        if not torch.isfinite(value).all().item():
            raise AssertionError(f"{variant} inference returned non-finite {field}")


@pytest.mark.skipif(
    not _RUN_COMPARE,
    reason="set DP_RUN_DPA4_COMPARE=1 to run the DPA4 inference comparison",
)
def test_compare_water_dpa4_pt_expt_vs_tilelang(tmp_path: Path) -> None:
    """Compare DeepMD baseline and TileLang DPA4 for energy, force, and latency."""
    _require_comparison_runtime()
    repo_root = Path(__file__).resolve().parents[4]
    example_dir = repo_root / "examples" / "water" / "dpa4" / "lmp"
    config_path = Path(
        os.environ.get("DP_DPA4_COMPARE_CONFIG", example_dir / "input.json")
    ).expanduser()
    checkpoint_path = Path(
        os.environ.get("DP_DPA4_COMPARE_CHECKPOINT", example_dir / "pretrained.pt")
    ).expanduser()
    geometry_path = Path(
        os.environ.get("DP_DPA4_COMPARE_GEOMETRY", example_dir / "water.lmp")
    ).expanduser()
    output_dir = Path(
        os.environ.get("DP_DPA4_COMPARE_DIR", tmp_path / "dpa4_compare")
    ).expanduser()
    for path, label in (
        (config_path, "config"),
        (checkpoint_path, "checkpoint"),
        (geometry_path, "geometry"),
    ):
        if not path.is_file():
            pytest.fail(f"DPA4 comparison {label} does not exist: {path}")

    device_name = os.environ.get("DP_DPA4_COMPARE_DEVICE", "cuda")
    device = torch.device(device_name)
    if device.type != "cuda":
        pytest.skip("DPA4 TileLang comparison currently requires a HIP/DCU CUDA device")
    warmup_steps = max(0, int(os.environ.get("DP_DPA4_COMPARE_WARMUP", "3")))
    repeat_steps = max(3, int(os.environ.get("DP_DPA4_COMPARE_REPEATS", "20")))
    rtol = float(os.environ.get("DP_DPA4_COMPARE_RTOL", "1.0e-3"))
    atol = float(os.environ.get("DP_DPA4_COMPARE_ATOL", "1.0e-4"))

    # Selectors are captured by DPA4 modules during construction. Use two
    # independent models so toggling flags never mutates a model being measured.
    with _inference_backend(tilelang_enabled=False):
        baseline_model, baseline_params = _load_model(
            config_path, checkpoint_path, device
        )
    with _inference_backend(tilelang_enabled=True):
        tilelang_model, tilelang_params = _load_model(
            config_path, checkpoint_path, device
        )
    if baseline_params != tilelang_params:
        raise AssertionError("baseline and TileLang model configurations differ")

    coordinates, atom_types, box = _read_water_lammps_data(geometry_path)
    descriptor_precision = str(
        baseline_params["descriptor"].get("precision", "float32")
    )
    input_dtype = torch.float64 if descriptor_precision == "float64" else torch.float32
    coordinates = coordinates.to(device=device, dtype=input_dtype)
    atom_types = atom_types.to(device=device)
    box = box.to(device=device, dtype=input_dtype)

    baseline_outputs, baseline_latency = _measure_variant(
        baseline_model,
        coordinates,
        atom_types,
        box,
        device,
        warmup_steps,
        repeat_steps,
    )
    tilelang_outputs, tilelang_latency = _measure_variant(
        tilelang_model,
        coordinates,
        atom_types,
        box,
        device,
        warmup_steps,
        repeat_steps,
    )
    _assert_finite_outputs("DeepMD-kit baseline", baseline_outputs)
    _assert_finite_outputs("TileLang DPA4", tilelang_outputs)

    energy_metrics = _error_metrics(
        baseline_outputs["energy"], tilelang_outputs["energy"]
    )
    force_metrics = _error_metrics(
        baseline_outputs["force"], tilelang_outputs["force"]
    )
    energy_within_tolerance = torch.allclose(
        tilelang_outputs["energy"],
        baseline_outputs["energy"],
        rtol=rtol,
        atol=atol,
    )
    force_within_tolerance = torch.allclose(
        tilelang_outputs["force"],
        baseline_outputs["force"],
        rtol=rtol,
        atol=atol,
    )

    speedup = baseline_latency["median_ms"] / tilelang_latency["median_ms"]
    report = {
        "comparison": "DeepMD-kit pt_expt baseline vs DP_TILELANG_DPA4",
        "model_config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "geometry": str(geometry_path),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "atom_count": int(coordinates.shape[1]),
        "dtype": str(coordinates.dtype),
        "warmup_steps": warmup_steps,
        "repeat_steps": repeat_steps,
        "tilelang_selectors": {
            "baseline": "all DP_TILELANG_* selectors disabled",
            "tilelang": "DP_TILELANG_DPA4=1",
        },
        "tolerance": {"rtol": rtol, "atol": atol},
        "numerical_agreement": {
            "energy": energy_metrics,
            "force": force_metrics,
            "within_tolerance": {
                "energy": energy_within_tolerance,
                "force": force_within_tolerance,
            },
            "note": (
                "Agreement against the DeepMD-kit path on this fixture; "
                "not labeled scientific accuracy."
            ),
        },
        "latency": {
            "deepmd_kit_baseline": baseline_latency,
            "tilelang_dpa4": tilelang_latency,
            "speedup_p50": speedup,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dpa4_comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    sys.stdout.write(f"DPA4 comparison report: {report_path}\n")
    sys.stdout.write(
        "Energy max_abs={:.6e}, force max_abs={:.6e}, force relative_l2={:.6e}\n".format(
            energy_metrics["max_abs"],
            force_metrics["max_abs"],
            force_metrics["relative_l2"],
        )
    )
    sys.stdout.write(
        "Latency p50: DeepMD-kit={:.3f} ms, TileLang={:.3f} ms, speedup={:.3f}x\n".format(
            baseline_latency["median_ms"],
            tilelang_latency["median_ms"],
            speedup,
        )
    )
    sys.stdout.flush()
    torch.testing.assert_close(
        tilelang_outputs["energy"],
        baseline_outputs["energy"],
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        tilelang_outputs["force"],
        baseline_outputs["force"],
        rtol=rtol,
        atol=atol,
    )
