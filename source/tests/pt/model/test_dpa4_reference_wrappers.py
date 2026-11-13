# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 wrappers stay on their reference path with retired TileLang flags set."""

import pytest
import torch

from deepmd.dpmodel.descriptor.dpa4_nn.radial import RadialBasis as RadialBasisDP
from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Linear as SO2LinearDP
from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
    WignerDCalculator as WignerDCalculatorDP,
)
from deepmd.pt_expt.descriptor.dpa4 import C3CutoffEnvelope as C3CutoffEnvelopePT
from deepmd.pt_expt.descriptor.dpa4 import RadialBasis as RadialBasisPT
from deepmd.pt_expt.descriptor.dpa4_nn.embedding import (
    GeometricInitialEmbedding as GeometricInitialEmbeddingPT,
)
from deepmd.pt_expt.descriptor.dpa4_nn.so2 import SO2Linear as SO2LinearPT
from deepmd.pt_expt.descriptor.dpa4_nn.wignerd import (
    WignerDCalculator as WignerDCalculatorPT,
)

_LEGACY_TILELANG_FLAGS = (
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


@pytest.fixture(autouse=True)
def _retired_tilelang_flags_are_ignored(monkeypatch):
    for name in _LEGACY_TILELANG_FLAGS:
        monkeypatch.setenv(name, "1")


def _make_radial_basis():
    reference = RadialBasisDP(
        rcut=6.0,
        basis_type="bessel",
        n_radial=6,
        precision="float32",
        exponent=7,
    )
    return reference, RadialBasisPT.deserialize(reference.serialize())


def _assert_value_and_input_grad(actual, expected, value):
    torch.testing.assert_close(actual, expected)
    (actual_grad,) = torch.autograd.grad(actual.sum(), value, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected.sum(), value)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_radial_basis_uses_reference_path():
    reference, wrapped = _make_radial_basis()
    wrapped.eval()
    distance = torch.tensor([[0.0], [0.1], [2.0]], requires_grad=True)
    _assert_value_and_input_grad(wrapped(distance), reference.call(distance), distance)


def test_c3_envelope_uses_reference_path():
    wrapped = C3CutoffEnvelopePT(rcut=6.0, exponent=5).eval()
    distance = torch.tensor([[0.0], [0.1], [5.5], [6.0], [7.0]], requires_grad=True)
    _assert_value_and_input_grad(wrapped(distance), wrapped.call(distance), distance)


def test_wigner_calculator_uses_reference_path():
    wrapped = WignerDCalculatorPT(lmax=1).eval()
    quaternion = torch.randn((5, 4), dtype=torch.float32, requires_grad=True)
    actual = wrapped.call(quaternion)
    expected = WignerDCalculatorDP(lmax=1).call(quaternion)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    grad_outputs = (torch.randn_like(actual[0]), torch.randn_like(actual[1]))
    (actual_grad,) = torch.autograd.grad(
        actual, quaternion, grad_outputs, retain_graph=True
    )
    (expected_grad,) = torch.autograd.grad(expected, quaternion, grad_outputs)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_so2_linear_uses_reference_path():
    kwargs = {
        "lmax": 1,
        "mmax": 1,
        "in_channels": 2,
        "out_channels": 3,
        "n_focus": 2,
        "precision": "float32",
        "mlp_bias": True,
        "seed": 19,
        "trainable": True,
    }
    wrapped = SO2LinearPT(**kwargs).eval()
    reference = SO2LinearDP(**kwargs)
    value = torch.randn((2, 5, 4, 2), dtype=torch.float32, requires_grad=True)
    _assert_value_and_input_grad(wrapped.call(value), reference.call(value), value)


def test_gie_fusion_does_not_read_retired_tilelang_flag():
    gie = GeometricInitialEmbeddingPT(lmax=1, channels=4).eval()
    assert not gie._can_fuse_scatter(torch.ones((2, 3), dtype=torch.float32))
