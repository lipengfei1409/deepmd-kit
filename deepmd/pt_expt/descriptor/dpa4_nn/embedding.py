# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch runtime bindings for DPA4 initial embeddings."""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.common import (
    get_xp_precision,
)
from deepmd.dpmodel.descriptor.dpa4_nn.embedding import (
    EnvironmentInitialEmbedding as EnvironmentInitialEmbeddingDP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.embedding import (
    GeometricInitialEmbedding as GeometricInitialEmbeddingDP,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)
from deepmd.pt_expt.kernels.utils import (
    cuda_infer_level,
    tilelang_operator_enabled,
)

from .edge_cache import (
    cached_edge_csr,
)


@torch_module
class EnvironmentInitialEmbedding(EnvironmentInitialEmbeddingDP):
    """Environment-seed embedding with an opt-in TileLang scatter stage."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tilelang_scatter_enabled = tilelang_operator_enabled(
            "DP_TILELANG_ENV_SEED_SCATTER"
        )

    def _aggregate_environment(
        self,
        r_tilde: torch.Tensor,
        features: torch.Tensor,
        edge_cache: Any,
        n_nodes: int,
    ) -> torch.Tensor:
        if self._tilelang_scatter_enabled:
            if self.training:
                raise RuntimeError("DP_TILELANG_ENV_SEED_SCATTER is inference-only")
            if (
                r_tilde.dtype != torch.float32
                or not r_tilde.is_cuda
                or torch.version.hip is None
            ):
                raise RuntimeError(
                    "TileLang environment-seed scatter requires float32 HIP/DCU tensors"
                )
            if (
                self.spin_flags is not None
                or edge_cache.edge_src_gate is not None
                or edge_cache.edge_mask is not None
            ):
                raise RuntimeError(
                    "TileLang environment-seed scatter does not support spin, masks or bridging"
                )
            from kernels.tilelang_ops import tilelang_environment_seed_scatter

            return tilelang_environment_seed_scatter(
                r_tilde.contiguous(),
                features.contiguous(),
                edge_cache.dst.contiguous(),
                edge_cache.inv_sqrt_deg.reshape(-1).contiguous(),
            )
        return super()._aggregate_environment(r_tilde, features, edge_cache, n_nodes)


register_dpmodel_mapping(
    EnvironmentInitialEmbeddingDP,
    lambda v: EnvironmentInitialEmbedding.deserialize(v.serialize()),
)


@torch_module
class GeometricInitialEmbedding(GeometricInitialEmbeddingDP):
    """Geometric initial embedding with an optional fused CUDA scatter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # === Fused message-and-scatter operator ===
        # The reference composition materializes the per-edge message, an
        # (E, D-1, C) tensor that dominates the cost of this module. The fused
        # operator keeps it in registers and reduces through the destination CSR.
        self._cuda_scatter = False
        self._tilelang_scatter_enabled = tilelang_operator_enabled(
            "DP_TILELANG_ZONAL_SCATTER"
        )
        # ``None`` keeps the runtime ``zonal_coupling.is_cuda`` dispatch; the
        # freeze pins it to the AOTI target because tracing always runs on CPU.
        self._force_fused_scatter: bool | None = None
        if (
            cuda_infer_level() >= 1
            and get_xp_precision(torch, self.precision) is torch.float32
        ):
            from deepmd.pt_expt.kernels.cuda.dpa4.zonal_scatter import (
                op_available,
                supported,
            )

            self._cuda_scatter = op_available() and supported(
                self.lmax, self.ebed_dim - 1, self.channels
            )

    def call(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self._tilelang_scatter_enabled:
            edge_cache = kwargs.get("edge_cache")
            if edge_cache is None:
                raise RuntimeError("TileLang zonal scatter requires an edge cache")
            if kwargs.get("spin_l1_message") is not None:
                raise RuntimeError(
                    "TileLang zonal scatter does not support spin messages"
                )
            if edge_cache.edge_src_gate is not None:
                raise RuntimeError(
                    "TileLang zonal scatter does not support bridging gates"
                )
            if edge_cache.csr_cache is None:
                raise RuntimeError(
                    "TileLang zonal scatter requires the standard edge cache"
                )
        return super().call(*args, **kwargs)

    def _can_fuse_scatter(self, zonal_coupling: torch.Tensor) -> bool:
        """Return whether a selected scatter implementation serves this call."""
        if self._tilelang_scatter_enabled:
            if self.training:
                raise RuntimeError("DP_TILELANG_ZONAL_SCATTER is inference-only")
            if (
                zonal_coupling.dtype != torch.float32
                or not zonal_coupling.is_cuda
                or torch.version.hip is None
            ):
                raise RuntimeError(
                    "TileLang zonal scatter requires float32 HIP/DCU tensors"
                )
            return True
        target_is_cuda = (
            zonal_coupling.is_cuda
            if self._force_fused_scatter is None
            else self._force_fused_scatter
        )
        return self._cuda_scatter and not self.training and target_is_cuda

    def forward_fused_scatter(
        self,
        n_nodes: int | torch.SymInt,
        edge_cache: Any,
        radial_feat: torch.Tensor,
        zonal_coupling: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build and reduce the geometric message with the fused CUDA operator.

        Parameters
        ----------
        n_nodes : int or torch.SymInt
            Number of nodes (nf * nloc).
        edge_cache : EdgeCache
            Per-edge cache supplying the destination endpoint, its CSR view and
            the smooth degree normalization.
        radial_feat : torch.Tensor
            Per-edge radial features with shape (E, lmax, C) for degrees
            1 to lmax.
        zonal_coupling : torch.Tensor
            Zonal coupling with shape (E, D-1).

        Returns
        -------
        torch.Tensor
            Initial features to add with shape (N, D, C), with l=0 zero.
        """
        if self._tilelang_scatter_enabled:
            from kernels.tilelang_ops import tilelang_zonal_scatter

            radial_slot = torch.as_tensor(
                self.radial_slot_index_for_row,
                dtype=torch.int64,
                device=radial_feat.device,
            )
            return tilelang_zonal_scatter(
                zonal_coupling.contiguous(),
                radial_feat.contiguous(),
                edge_cache.dst.contiguous(),
                radial_slot,
                edge_cache.inv_sqrt_deg.reshape(-1).contiguous(),
            )

        from deepmd.pt_expt.kernels.cuda.dpa4.zonal_scatter import (
            zonal_scatter,
        )

        # === Step 1. Destination CSR, shared with every other edge consumer ===
        order, row_ptr = cached_edge_csr(edge_cache, "dst", n_nodes)

        # === Step 2. Fused message build, reduction, padding and normalization ===
        # The operator emits the packed node layout already normalized, so the
        # scalar row and the degree scaling cost no extra pass. The scaling is
        # differentiated: the smooth degree is a sum over the cutoff envelope
        # and carries a gradient back to the geometry.
        return zonal_scatter(
            zonal_coupling.contiguous(),
            radial_feat.contiguous(),
            edge_cache.dst,
            order,
            row_ptr,
            edge_cache.inv_sqrt_deg.reshape(-1),
            n_nodes,
        )  # (N, D, C)
