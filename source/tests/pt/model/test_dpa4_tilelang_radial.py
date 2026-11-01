# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the opt-in TileLang radial-basis integration."""

import pytest
import torch

from deepmd.dpmodel.descriptor.dpa4_nn.radial import RadialBasis as RadialBasisDP
from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Linear as SO2LinearDP
from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
    WignerDCalculator as WignerDCalculatorDP,
)
from deepmd.pt_expt.descriptor.dpa4 import (
    C3CutoffEnvelope as C3CutoffEnvelopePT,
)
from deepmd.pt_expt.descriptor.dpa4 import DescrptDPA4 as DescrptDPA4PT
from deepmd.pt_expt.descriptor.dpa4 import (
    GridProduct as GridProductPT,
)
from deepmd.pt_expt.descriptor.dpa4 import (
    RadialBasis as RadialBasisPT,
)
from deepmd.pt_expt.descriptor.dpa4_nn.embedding import (
    EnvironmentInitialEmbedding as EnvironmentInitialEmbeddingPT,
)
from deepmd.pt_expt.descriptor.dpa4_nn.embedding import (
    GeometricInitialEmbedding as GeometricInitialEmbeddingPT,
)
from deepmd.pt_expt.descriptor.dpa4_nn.so2 import SO2Convolution as SO2ConvolutionPT
from deepmd.pt_expt.descriptor.dpa4_nn.so2 import SO2Linear as SO2LinearPT
from deepmd.pt_expt.descriptor.dpa4_nn.wignerd import (
    WignerDCalculator as WignerDCalculatorPT,
)


@pytest.fixture(autouse=True)
def _disable_suite_tilelang_switch(monkeypatch):
    monkeypatch.delenv("DP_TILELANG_DPA4", raising=False)


def _make_radial_basis():
    reference = RadialBasisDP(
        rcut=6.0,
        basis_type="bessel",
        n_radial=6,
        precision="float32",
        exponent=7,
    )
    return reference, RadialBasisPT.deserialize(reference.serialize())


def test_radial_basis_wrapper_preserves_default_pytorch_path(monkeypatch):
    monkeypatch.delenv("DP_TILELANG_RADIAL_BASIS", raising=False)
    reference, wrapped = _make_radial_basis()
    wrapped.eval()
    distance = torch.tensor([[0.0], [0.1], [2.0]], requires_grad=True)

    actual = wrapped(distance)
    expected = reference.call(distance)
    torch.testing.assert_close(actual, expected)
    (actual_grad,) = torch.autograd.grad(actual.sum(), distance)
    (expected_grad,) = torch.autograd.grad(expected.sum(), distance)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_enabled_tilelang_path_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_RADIAL_BASIS", "1")
    _, wrapped = _make_radial_basis()
    wrapped.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        wrapped(torch.ones((2, 1), dtype=torch.float32))


def test_enabled_tilelang_path_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_RADIAL_BASIS", "1")
    _, wrapped = _make_radial_basis()
    wrapped.eval()
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        wrapped(torch.ones((2, 1), dtype=torch.float32))


def test_c3_envelope_wrapper_preserves_default_pytorch_path(monkeypatch):
    monkeypatch.delenv("DP_TILELANG_C3_ENVELOPE", raising=False)
    reference = C3CutoffEnvelopePT(rcut=6.0, exponent=5)
    reference.eval()
    distance = torch.tensor([[0.0], [0.1], [5.5], [6.0], [7.0]], requires_grad=True)

    actual = reference(distance)
    expected = reference.call(distance)
    torch.testing.assert_close(actual, expected)
    (actual_grad,) = torch.autograd.grad(actual.sum(), distance, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected.sum(), distance)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_enabled_c3_envelope_path_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_C3_ENVELOPE", "1")
    envelope = C3CutoffEnvelopePT(rcut=6.0, exponent=5)
    envelope.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        envelope(torch.ones((2, 1), dtype=torch.float32))


def test_enabled_c3_envelope_path_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_C3_ENVELOPE", "1")
    envelope = C3CutoffEnvelopePT(rcut=6.0, exponent=5)
    envelope.eval()
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        envelope(torch.ones((2, 1), dtype=torch.float32))


def test_wigner_l1_wrapper_preserves_default_pytorch_path(monkeypatch):
    monkeypatch.delenv("DP_TILELANG_WIGNER_L1", raising=False)
    wrapped = WignerDCalculatorPT(lmax=1)
    wrapped.eval()
    quaternion = torch.randn((5, 4), dtype=torch.float32, requires_grad=True)

    actual = wrapped.call(quaternion)
    expected = WignerDCalculatorDP(lmax=1).call(quaternion)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    grad_outputs = (torch.randn_like(actual[0]), torch.randn_like(actual[1]))
    (actual_grad,) = torch.autograd.grad(actual, quaternion, grad_outputs)
    (expected_grad,) = torch.autograd.grad(expected, quaternion, grad_outputs)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_enabled_wigner_l1_path_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_WIGNER_L1", "1")
    wrapped = WignerDCalculatorPT(lmax=1)
    wrapped.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        wrapped.call(torch.ones((2, 4), dtype=torch.float32))


def test_enabled_wigner_l1_path_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_WIGNER_L1", "1")
    wrapped = WignerDCalculatorPT(lmax=1)
    wrapped.eval()
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        wrapped.call(torch.ones((2, 4), dtype=torch.float32))


def test_enabled_gie_scatter_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_ZONAL_SCATTER", "1")
    gie = GeometricInitialEmbeddingPT(lmax=1, channels=4)
    gie.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        gie._can_fuse_scatter(torch.ones((2, 3), dtype=torch.float32))


def test_enabled_gie_scatter_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_ZONAL_SCATTER", "1")
    gie = GeometricInitialEmbeddingPT(lmax=1, channels=4)
    gie.eval()
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        gie._can_fuse_scatter(torch.ones((2, 3), dtype=torch.float32))


def test_so2_linear_wrapper_preserves_default_pytorch_path(monkeypatch):
    monkeypatch.delenv("DP_TILELANG_SO2_LINEAR_L1", raising=False)
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
    x = torch.randn((2, 5, 4, 2), dtype=torch.float32, requires_grad=True)
    actual = wrapped.call(x)
    expected = reference.call(x)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    (actual_grad,) = torch.autograd.grad(actual, x, upstream, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected, x, upstream)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_enabled_so2_linear_path_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_SO2_LINEAR_L1", "1")
    layer = SO2LinearPT(
        lmax=1,
        mmax=1,
        in_channels=2,
        out_channels=3,
        n_focus=1,
        precision="float32",
        mlp_bias=False,
        seed=19,
        trainable=True,
    )
    x = torch.ones((1, 2, 8), dtype=torch.float32)
    weight = torch.ones((8, 1, 12), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="inference-only"):
        layer._block_diagonal_matmul(x, weight)


def test_enabled_so2_linear_path_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_SO2_LINEAR_L1", "1")
    layer = SO2LinearPT(
        lmax=1,
        mmax=1,
        in_channels=2,
        out_channels=3,
        n_focus=1,
        precision="float32",
        mlp_bias=False,
        seed=19,
        trainable=True,
    ).eval()
    x = torch.ones((1, 2, 8), dtype=torch.float32)
    weight = torch.ones((8, 1, 12), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        layer._block_diagonal_matmul(x, weight)


def test_enabled_so2_rotation_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_SO2_ROTATION_L1", "1")
    conv = SO2ConvolutionPT(
        lmax=1,
        mmax=1,
        channels=4,
        mixing_layers=0,
        n_atten_head=0,
        precision="float32",
        seed=7,
        trainable=True,
    )
    conv.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        conv._rotate_to_local(torch.ones((2, 4, 4), dtype=torch.float32), None)


def test_enabled_so2_rotation_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_SO2_ROTATION_L1", "1")
    conv = SO2ConvolutionPT(
        lmax=1,
        mmax=1,
        channels=4,
        mixing_layers=0,
        n_atten_head=0,
        precision="float32",
        seed=7,
        trainable=True,
    ).eval()
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        conv._rotate_to_local(torch.ones((2, 4, 4), dtype=torch.float32), None)


def test_enabled_attention_softmax_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_ATTENTION_SOFTMAX", "1")
    conv = SO2ConvolutionPT(
        lmax=1,
        mmax=1,
        channels=4,
        mixing_layers=0,
        n_atten_head=1,
        precision="float32",
        seed=7,
        trainable=True,
    )
    conv.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        conv._attention_softmax(torch.ones((2, 1, 1), dtype=torch.float32), None, 2)


def test_enabled_attention_softmax_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_ATTENTION_SOFTMAX", "1")
    conv = SO2ConvolutionPT(
        lmax=1,
        mmax=1,
        channels=4,
        mixing_layers=0,
        n_atten_head=1,
        precision="float32",
        seed=7,
        trainable=True,
    ).eval()
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        conv._attention_softmax(torch.ones((2, 1, 1), dtype=torch.float32), None, 2)


def test_enabled_attention_aggregate_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_ATTENTION_AGGREGATE", "1")
    conv = SO2ConvolutionPT(
        lmax=1,
        mmax=1,
        channels=4,
        mixing_layers=0,
        n_atten_head=1,
        precision="float32",
        seed=7,
        trainable=True,
    )
    conv.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        conv._attention_weighted_reduce(None, None, None, None, 2)


def test_enabled_attention_aggregate_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_ATTENTION_AGGREGATE", "1")
    conv = SO2ConvolutionPT(
        lmax=1,
        mmax=1,
        channels=4,
        mixing_layers=0,
        n_atten_head=1,
        precision="float32",
        seed=7,
        trainable=True,
    ).eval()
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        conv._attention_weighted_reduce(
            torch.ones((2, 4, 1, 1, 4), dtype=torch.float32),
            torch.ones((2, 1, 1), dtype=torch.float32),
            torch.ones((2, 1, 1), dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.int64),
            2,
        )


def test_enabled_so2_radial_scale_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_SO2_RADIAL_SCALE", "1")
    conv = SO2ConvolutionPT(
        lmax=1,
        mmax=1,
        channels=4,
        mixing_layers=0,
        n_atten_head=0,
        precision="float32",
        seed=7,
        trainable=True,
    )
    conv.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        conv._apply_radial_scale(torch.ones((2, 4, 4)), torch.ones((2, 4, 4)))


def test_enabled_so2_radial_scale_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_SO2_RADIAL_SCALE", "1")
    conv = SO2ConvolutionPT(
        lmax=1,
        mmax=1,
        channels=4,
        mixing_layers=0,
        n_atten_head=0,
        precision="float32",
        seed=7,
        trainable=True,
    ).eval()
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        conv._apply_radial_scale(torch.ones((2, 4, 4)), torch.ones((2, 4, 4)))


def test_enabled_env_seed_scatter_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_ENV_SEED_SCATTER", "1")
    embedding = EnvironmentInitialEmbeddingPT(
        ntypes=2,
        n_radial=4,
        channels=4,
        embed_dim=8,
        axis_dim=4,
        type_dim=2,
        hidden_dim=8,
        precision="float32",
        seed=7,
        trainable=True,
    )
    embedding.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        embedding._aggregate_environment(
            torch.ones((2, 4), dtype=torch.float32),
            torch.ones((2, 8), dtype=torch.float32),
            None,
            2,
        )


def test_enabled_env_seed_scatter_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_ENV_SEED_SCATTER", "1")
    embedding = EnvironmentInitialEmbeddingPT(
        ntypes=2,
        n_radial=4,
        channels=4,
        embed_dim=8,
        axis_dim=4,
        type_dim=2,
        hidden_dim=8,
        precision="float32",
        seed=7,
        trainable=True,
    ).eval()
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        embedding._aggregate_environment(
            torch.ones((2, 4), dtype=torch.float32),
            torch.ones((2, 8), dtype=torch.float32),
            None,
            2,
        )


def test_enabled_grid_product_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_GRID_PRODUCT", "1")
    grid_product = GridProductPT()
    grid_product.train()

    def identity(value):
        return value

    with pytest.raises(RuntimeError, match="inference-only"):
        grid_product.call(
            torch.ones((2, 3), dtype=torch.float32),
            torch.ones((2, 3), dtype=torch.float32),
            torch.empty((0,)),
            to_grid=identity,
            from_grid=identity,
        )


def test_enabled_grid_product_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_GRID_PRODUCT", "1")
    grid_product = GridProductPT().eval()

    def identity(value):
        return value

    with pytest.raises(RuntimeError, match="HIP/DCU"):
        grid_product.call(
            torch.ones((2, 3), dtype=torch.float32),
            torch.ones((2, 3), dtype=torch.float32),
            torch.empty((0,)),
            to_grid=identity,
            from_grid=identity,
        )


def _make_small_dpa4_descriptor():
    return DescrptDPA4PT(
        ntypes=2,
        sel=[4],
        channels=4,
        n_radial=2,
        radial_mlp=[4],
        lmax=1,
        mmax=1,
        n_blocks=1,
        mixing_layers=0,
        n_atten_head=1,
        radial_so2_mode="none",
        precision="float32",
        use_amp=False,
    )


def test_edge_geometry_switch_keeps_default_dispatch(monkeypatch):
    monkeypatch.delenv("DP_TILELANG_EDGE_GEOMETRY", raising=False)
    descriptor = _make_small_dpa4_descriptor()
    assert descriptor._tilelang_edge_geometry_fn is None


def test_enabled_edge_geometry_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_EDGE_GEOMETRY", "1")
    descriptor = _make_small_dpa4_descriptor()
    edge_vec = torch.ones((2, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="inference-only"):
        descriptor._tilelang_edge_geometry_fn(edge_vec, 1.0e-7)


def test_enabled_edge_geometry_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_EDGE_GEOMETRY", "1")
    descriptor = _make_small_dpa4_descriptor().eval()
    edge_vec = torch.ones((2, 3), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        descriptor._tilelang_edge_geometry_fn(edge_vec, 1.0e-7)


def test_enabled_edge_quaternion_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_EDGE_QUATERNION", "1")
    descriptor = _make_small_dpa4_descriptor()
    edge_vec = torch.ones((2, 3), dtype=torch.float32)
    edge_len = torch.ones((2, 1), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="inference-only"):
        descriptor._tilelang_edge_quaternion_fn(edge_vec, edge_len, 1.0e-7)


def test_enabled_edge_quaternion_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_EDGE_QUATERNION", "1")
    descriptor = _make_small_dpa4_descriptor().eval()
    edge_vec = torch.ones((2, 3), dtype=torch.float32)
    edge_len = torch.ones((2, 1), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        descriptor._tilelang_edge_quaternion_fn(edge_vec, edge_len, 1.0e-7)


def _make_env_seed_embedding():
    return EnvironmentInitialEmbeddingPT(
        ntypes=2,
        n_radial=4,
        channels=4,
        embed_dim=8,
        axis_dim=4,
        type_dim=2,
        hidden_dim=8,
        precision="float32",
        seed=7,
        trainable=True,
    )


def test_enabled_env_seed_matrix_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_ENV_SEED_MATRIX", "1")
    embedding = _make_env_seed_embedding()
    with pytest.raises(RuntimeError, match="inference-only"):
        embedding._build_r_tilde(torch.ones((2, 3)), torch.ones((2, 1)))


def test_enabled_env_seed_matrix_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_ENV_SEED_MATRIX", "1")
    embedding = _make_env_seed_embedding().eval()
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        embedding._build_r_tilde(torch.ones((2, 3)), torch.ones((2, 1)))


def test_enabled_env_seed_gram_rejects_training_mode(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_ENV_SEED_GRAM", "1")
    embedding = _make_env_seed_embedding()
    with pytest.raises(RuntimeError, match="inference-only"):
        embedding._environment_gram(torch.ones((2, 4, 8)))


def test_enabled_env_seed_gram_does_not_fall_back_on_cpu(monkeypatch):
    monkeypatch.setenv("DP_TILELANG_ENV_SEED_GRAM", "1")
    embedding = _make_env_seed_embedding().eval()
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        embedding._environment_gram(torch.ones((2, 4, 8)))
