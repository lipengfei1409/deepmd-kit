# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.common import (
    get_xp_precision,
)
from deepmd.dpmodel.descriptor.dpa4 import DescrptDPA4 as DescrptDPA4DP
from deepmd.dpmodel.descriptor.dpa4_nn.activation import SwiGLU as SwiGLUDP
from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import GridProduct as GridProductDP
from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
    BridgingSwitch as BridgingSwitchDP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
    C3CutoffEnvelope as C3CutoffEnvelopeDP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
    InnerClamp as InnerClampDP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
    RadialBasis as RadialBasisDP,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.kernels.utils import (
    cuda_infer_level,
    tilelang_dpa4_enabled,
    tilelang_operator_enabled,
    use_amp_infer,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@torch_module
class SwiGLU(SwiGLUDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


# SwiGLU is parameter-free (no serialize); rebuild fresh.
register_dpmodel_mapping(SwiGLUDP, lambda v: SwiGLU())


@torch_module
class C3CutoffEnvelope(C3CutoffEnvelopeDP):
    """C3 cutoff envelope with an opt-in standalone TileLang inference path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tilelang_infer_enabled = tilelang_operator_enabled(
            "DP_TILELANG_C3_ENVELOPE"
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if not self._tilelang_infer_enabled:
            return self.call(r)
        if self.training:
            raise RuntimeError("DP_TILELANG_C3_ENVELOPE is inference-only")
        if r.dtype != torch.float32 or not r.is_cuda or torch.version.hip is None:
            raise RuntimeError(
                "TileLang C3 cutoff envelope requires float32 HIP/DCU tensors"
            )

        from kernels.tilelang_ops import tilelang_c3_cutoff_envelope

        return tilelang_c3_cutoff_envelope(r, self.rcut, self.p)


@torch_module
class RadialBasis(RadialBasisDP):
    """Radial basis with an opt-in, standalone TileLang inference path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tilelang_infer_enabled = tilelang_operator_enabled(
            "DP_TILELANG_RADIAL_BASIS"
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if not self._tilelang_infer_enabled:
            return self.call(r)
        if self.training:
            raise RuntimeError("DP_TILELANG_RADIAL_BASIS is inference-only")
        if self.basis_family != "bessel":
            raise RuntimeError(
                "TileLang radial basis currently supports only Bessel basis"
            )
        if r.dtype != torch.float32 or not r.is_cuda or torch.version.hip is None:
            raise RuntimeError("TileLang radial basis requires float32 HIP/DCU tensors")

        from kernels.tilelang_ops import tilelang_bessel_radial_basis

        # Keep the cutoff envelope as an ordinary separate PyTorch op; this
        # TileLang operator computes only the raw Bessel basis.
        radial = tilelang_bessel_radial_basis(r, self.adam_freqs)
        if self.envelope is not None:
            radial = radial * self.envelope(r)
        return radial


register_dpmodel_mapping(
    RadialBasisDP,
    lambda v: RadialBasis.deserialize(v.serialize()),
)


@torch_module
class InnerClamp(InnerClampDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


# InnerClamp/BridgingSwitch are parameter-free (scalar bridging radii only,
# no serialize()); rebuild fresh from the stored constructor arguments.
register_dpmodel_mapping(
    InnerClampDP,
    lambda v: InnerClamp(v.r_inner, v.r_outer),
)


@torch_module
class BridgingSwitch(BridgingSwitchDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


register_dpmodel_mapping(
    BridgingSwitchDP,
    lambda v: BridgingSwitch(v.r_inner, v.r_outer),
)


# C3CutoffEnvelope carries only scalar configuration (cutoff radius and
# polynomial exponent) and holds no trainable arrays, so it implements no
# serialize()/deserialize() that the generic auto-wrap path relies on; rebuild
# it directly from the stored constructor arguments (``p`` is the exponent).
register_dpmodel_mapping(
    C3CutoffEnvelopeDP,
    lambda v: C3CutoffEnvelope(v.rcut, v.p, precision=v.precision),
)


@torch_module
class GridProduct(GridProductDP):
    """Pointwise S2 grid product with an opt-in standalone TileLang kernel."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tilelang_infer_enabled = tilelang_operator_enabled(
            "DP_TILELANG_GRID_PRODUCT"
        )

    def call(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        scalar_pair: torch.Tensor,
        *,
        to_grid: Any,
        from_grid: Any,
        pair_grid: Any = None,
        project_pair: Any = None,
        scalar_product: Any = None,
    ) -> torch.Tensor:
        if not self._tilelang_infer_enabled:
            return super().call(
                left,
                right,
                scalar_pair,
                to_grid=to_grid,
                from_grid=from_grid,
                pair_grid=pair_grid,
                project_pair=project_pair,
                scalar_product=scalar_product,
            )
        if self.training:
            raise RuntimeError("DP_TILELANG_GRID_PRODUCT is inference-only")
        if pair_grid is not None or scalar_product is not None:
            raise RuntimeError(
                "TileLang grid product does not support fused grid callbacks"
            )
        if project_pair is None:
            left_grid, right_grid = to_grid(left), to_grid(right)
        else:
            left_grid, right_grid = project_pair(left, right)
        if (
            left_grid.dtype != torch.float32
            or not left_grid.is_cuda
            or torch.version.hip is None
        ):
            raise RuntimeError("TileLang grid product requires float32 HIP/DCU tensors")

        from kernels.tilelang_ops import tilelang_grid_product

        return from_grid(tilelang_grid_product(left_grid, right_grid))


# GridProduct is a parameter-free quadratic grid product with no constructor
# arguments and no serialize()/deserialize(); rebuild a fresh instance.
register_dpmodel_mapping(GridProductDP, lambda v: GridProduct())


# ---------------------------------------------------------------------------
# Trainable-weight promotion
#
# ``dpmodel_setattr`` registers every numpy attribute as a torch *buffer*, so
# the auto-wrapped dpa4_nn sub-modules would otherwise expose their trainable
# weights as non-trainable buffers (no autograd, invisible to the optimizer).
# The table below lists, per dpmodel class name, the attributes that are
# ``torch.nn.Parameter`` in the reference pt SeZM implementation
# (deepmd/pt/model/descriptor/sezm_nn).  ``_promote_trainable_tree`` walks the
# fully-built module tree and re-registers those buffers as Parameters.
#
# Constant float buffers (e.g. ``balance_weight``, ``rotate_inv_rescale_full``,
# ``mean``/``stddev``) are intentionally NOT listed: they are buffers in pt
# too.  Lists of weights (e.g. ``SO2Linear.weight_m``) are already converted
# to trainable ``ParameterList`` by ``_try_convert_list``.
# ---------------------------------------------------------------------------
_TRAINABLE_ATTRS: dict[str, tuple[str, ...]] = {
    # dpa4_nn.norm
    "RMSNorm": ("adam_scale",),
    "EquivariantRMSNorm": ("adam_scale", "bias"),
    "ReducedEquivariantRMSNorm": ("adam_scale", "bias0"),
    "ScalarRMSNorm": ("adam_scale",),
    # dpa4_nn.radial
    "RadialBasis": ("adam_freqs",),
    # dpa4_nn.so3
    "SO3Linear": ("weight", "bias"),
    "FocusLinear": ("weight", "bias"),
    "ChannelLinear": ("weight", "bias"),
    # dpa4_nn.so2
    "SO2Linear": ("weight_m0", "bias0"),
    "DynamicRadialDegreeMixer": ("weight", "channel_basis"),
    "SO2Convolution": (
        "adamw_attn_logit_w",
        "adamw_attn_z_bias_raw",
        "adamw_attn_gate_w",
        "adamw_focus_compete_w",
        "focus_compete_bias",
    ),
    # dpa4_nn.embedding
    "SeZMTypeEmbedding": ("adam_type_embedding",),
    # dpa4_nn.embedding (native spin): these are nn.Parameter in pt but land as
    # numpy->buffer in dpmodel; mag_layer1/2 are NativeLayer and auto-promote,
    # and _promote_trainable skips a missing buffer, so no-spin configs (where
    # spin_scale is absent) stay safe.
    "SpinEmbedding": ("adam_spin_vec_weight", "adam_spin_nbr_weight"),
    "EnvironmentInitialEmbedding": ("spin_scale",),
    # dpa4_nn.attn_res
    "DepthAttnRes": ("adamw_pseudo_query",),
    # dpa4_nn.grid_net (residual_scale is None when disabled; _promote_trainable
    # skips the missing buffer, so listing both concrete subclasses is safe)
    "S2GridNet": ("residual_scale",),
    "SO3GridNet": ("residual_scale",),
    # dpa4_nn.grid_net frame mixing, built only by ``mode="cross"`` grid nets.
    # Unlike the surrounding projections these are plain numpy arrays rather
    # than NativeLayer objects, so they need an explicit entry here.
    "FrameExpand": ("weight",),
    "FrameContract": ("weight",),
    # descriptor-level FiLM strengths
    "DescrptDPA4": ("film_scale_strength_log", "film_shift_strength_log"),
}


def _promote_trainable(module: torch.nn.Module, names: tuple[str, ...]) -> None:
    """Re-register the given float buffers of *module* as Parameters."""
    trainable = bool(getattr(module, "trainable", True))
    for name in names:
        buf = module._buffers.get(name)
        if buf is None or not buf.is_floating_point():
            continue
        del module._buffers[name]
        setattr(module, name, torch.nn.Parameter(buf, requires_grad=trainable))


def _promote_trainable_tree(module: torch.nn.Module) -> torch.nn.Module:
    """Promote trainable buffers to Parameters across the whole module tree.

    Must run after the tree is fully built (post ``__init__`` /
    ``deserialize``): dpmodel deserialize may assign numpy arrays onto nested
    attributes, which ``dpmodel_setattr`` would re-register as buffers.
    """
    for sub in module.modules():
        names = _TRAINABLE_ATTRS.get(type(sub).__name__)
        if names is not None:
            _promote_trainable(sub, names)
    # Freeze every Parameter under a ``trainable=False`` module.  This covers
    # parameters that exist regardless of the promotion table above, e.g. the
    # ``SO2Linear.weight_m`` list, which ``_try_convert_list`` converts to a
    # ParameterList with ``requires_grad=True`` unconditionally.
    for sub in module.modules():
        if getattr(sub, "trainable", True) is False:
            for p in sub.parameters(recurse=True):
                p.requires_grad_(False)
    return module


@BaseDescriptor.register("SeZM")
@BaseDescriptor.register("sezm")
@BaseDescriptor.register("DPA4")
@BaseDescriptor.register("dpa4")
@torch_module
class DescrptDPA4(DescrptDPA4DP):
    _update_sel_cls = UpdateSel

    def adam_route_patterns(self) -> list[str]:
        """
        Name patterns, relative to the descriptor, of the tensors that take the
        AdamW path under HybridMuon: the first layer of the radial embedding and
        the radial projection of the environment seed, which read the radial
        basis and whose rows for rarely visited separations receive almost no
        gradient.
        """
        return [
            "radial_embedding.net.0.",
            "env_seed_embedding.rbf_proj_layer1.",
        ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tilelang_dpa4_enabled = tilelang_dpa4_enabled()
        self._tilelang_edge_geometry_enabled = tilelang_operator_enabled(
            "DP_TILELANG_EDGE_GEOMETRY"
        )
        self._tilelang_edge_quaternion_enabled = tilelang_operator_enabled(
            "DP_TILELANG_EDGE_QUATERNION"
        )
        self._tilelang_edge_geometry_fn = (
            self._run_tilelang_edge_geometry
            if self._tilelang_edge_geometry_enabled
            else None
        )
        self._tilelang_edge_quaternion_fn = (
            self._run_tilelang_edge_quaternion
            if self._tilelang_edge_quaternion_enabled
            else None
        )
        # The fused convolution paths consume only the three structural rows of
        # each Wigner degree block. Source-gated attention bypasses that fused
        # convolution, so its dense per-edge rotations remain available.
        self._wigner_free_conv = (
            self.bridging_switch is None
            and bool(self.blocks)
            and all(
                getattr(block.so2_conv, "_cuda_conv_fn", None) is not None
                and not block.so2_conv._cuda_conv_fn._compete
                for block in self.blocks
            )
        )
        self._packed_wigner_train = bool(self.blocks) and all(
            getattr(block.so2_conv, "_cuda_value_train", None) is not None
            and block.so2_conv._flash_atten_fn is not None
            and block.so2_conv._flash_atten_trains
            for block in self.blocks
        )

        # The envelope and the radial basis are both functions of the pair
        # distance and are cheap enough that the compiler inlines them into
        # every consumer and re-evaluates them there. Behind an operator
        # boundary the chain runs once per step.
        self._cuda_radial_fn = None
        self._cuda_wigner_fn = None
        if cuda_infer_level() >= 1:
            from deepmd.pt_expt.kernels.cuda.dpa4.edge_radial import (
                make_cuda_edge_radial,
            )
            from deepmd.pt_expt.kernels.cuda.dpa4.wigner_dense import (
                make_cuda_wigner_dense,
            )

            self._cuda_radial_fn = make_cuda_edge_radial(
                self.edge_envelope, self.radial_basis
            )
            # The dense Wigner pair otherwise costs five full-size passes
            # over the (E, D, D) tensors; the fused build pays only the
            # output writes.
            self._cuda_wigner_fn = make_cuda_wigner_dense(
                self.mp_init_lmax,
                get_xp_precision(torch, self.compute_precision),
            )

        # Persisted graph-routing knob (first-class training configuration):
        # ``disable_graph_lower()`` used to flip only the plain dpmodel bool,
        # which a Trainer checkpoint restart silently reset (the fresh model
        # is rebuilt from config before ``load_state_dict``, and neither the
        # state-dict keys nor ``_extra_state.model_params`` carried the
        # choice) -- on a binding-sel system that switched the training
        # equation and gradients without warning.  A persistent buffer rides
        # every pt_expt state_dict, so save/restart round-trips it.
        torch.nn.Module.register_buffer(
            self,
            "graph_lower_disabled",
            torch.zeros((), dtype=torch.bool, device="cpu"),
        )
        # Persisted descriptor version, for the same reason: pt_expt rebuilds
        # the module from config before loading, so without a buffer every
        # checkpoint would come back claiming the semantics of the running
        # code and silently skip ``_migrate_variables``.
        torch.nn.Module.register_buffer(
            self,
            "version_tensor",
            torch.tensor(self.version, dtype=torch.float64, device="cpu"),
        )
        self.use_amp_infer = use_amp_infer()
        _promote_trainable_tree(self)

    def _shared_wigner_runs(self, edge_cache: Any, lmax: int) -> torch.Tensor | None:
        """
        Zonal coupling taken from the packed runs the convolution already builds.

        The fused convolution stages a packed block-diagonal Wigner run per
        edge whose degree-``l`` ``m = 0`` row occupies entries ``l ** 2`` to
        ``(l + 1) ** 2``. That is the same quantity as
        ``Dt_full[:, row(l, m), col(l, 0)]``, so degrees ``1..lmax`` are one
        contiguous slice and the rotation algebra runs once per step instead of
        twice. The runs are cached on the edge cache, so whichever consumer
        comes first pays for them.

        Parameters
        ----------
        edge_cache : EdgeCache
            The step's edge feature cache.
        lmax : int
            Highest degree the coupling must cover.

        Returns
        -------
        torch.Tensor or None
            Coupling with shape ``(E, (lmax + 1) ** 2 - 1)``, or ``None`` when
            no convolution supplies runs of at least this degree.
        """
        if edge_cache.csr_cache is None:
            return None
        if self.training:
            if not self._packed_wigner_train:
                return None
            fused = self.blocks[0].so2_conv._cuda_value_train
        else:
            if not self._wigner_free_conv:
                return None
            fused = self.blocks[0].so2_conv._cuda_conv_fn
        if fused is None or lmax > self.lmax:
            return None
        return fused.edge_runs(edge_cache)[:, 1 : (lmax + 1) ** 2]

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA4":
        # deserialize assigns numpy arrays after __init__, which demotes
        # promoted Parameters back to buffers; re-promote at the end.
        obj = super().deserialize(data)
        # The buffer carries the version of the restored variables, not the
        # version the fresh construction started from.
        obj.version_tensor.fill_(obj.version)
        return _promote_trainable_tree(obj)

    def _in_training_mode(self) -> bool:
        """Torch runtime hook for the training-only random local-Z roll.

        Overrides the dpmodel default (``False``) with the torch module's
        ``training`` flag, restoring pt's ``random_gamma=self.random_gamma
        and self.training`` semantics: train-mode forwards draw a fresh
        gamma per call, eval/export forwards fix gamma (the export path
        calls ``model.eval()`` before tracing).
        """
        return bool(self.training)

    def _gate_partial_exchange(
        self,
        partials: torch.Tensor,
        comm_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Reverse-accumulate ghost partials to owners, then broadcast back.

        ``border_op_backward`` sums each ghost row into its owner across
        ranks and zeroes the ghost rows; ``border_op`` refills them with the
        completed owner values. Both ops carry autograd (the two are
        transposes), so gate gradients cross ranks (issue #5906).

        Parameters
        ----------
        partials
            (n_nodes, 2) float tensor of [log_eta, zero_count] partials.
        comm_dict
            The border-exchange control tensors.

        Returns
        -------
        torch.Tensor
            The globally completed (n_nodes, 2) tensor.
        """
        # border_op exchanges rows by raw pointer arithmetic; a strided
        # view would corrupt the exchange.
        p = partials.contiguous()
        comm_args = (
            comm_dict["send_list"],
            comm_dict["send_proc"],
            comm_dict["recv_proc"],
            comm_dict["send_num"],
            comm_dict["recv_num"],
        )
        tail = (
            comm_dict["communicator"],
            comm_dict["nlocal"],
            comm_dict["nghost"],
        )
        p = torch.ops.deepmd_export.border_op_backward(*comm_args, p, *tail)
        p = torch.ops.deepmd_export.border_op(*comm_args, p, *tail)
        return p

    def disable_graph_lower(self) -> None:
        """Persisted variant of the dpmodel escape hatch (see base class).

        The buffer (and the routing bool) are PER-TASK state: multi-task
        ``share_params`` shares network submodules, not this buffer, so
        disabling the graph lower on one task branch does not propagate to
        branches sharing the same descriptor weights -- each branch owns
        its routing decision.
        """
        super().disable_graph_lower()
        self.graph_lower_disabled.fill_(True)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Back-compat: checkpoints written before the knob was persisted lack
        # the buffer; default to the fresh module's value (graph enabled)
        # instead of failing the strict load.
        key = prefix + "graph_lower_disabled"
        if key not in state_dict:
            state_dict[key] = self.graph_lower_disabled.detach().clone()
        else:
            # Re-sync the dpmodel-side routing bool from the RESTORED value
            # here, at load time, where the incoming tensor is real.  The
            # routing predicate itself must stay a plain python bool:
            # ``uses_graph_lower()`` runs inside traced forwards (the dense
            # adapter gate), and reading the buffer there would emit a
            # data-dependent ``bool(FakeTensor)`` guard that breaks
            # torch.export (GuardOnDataDependentSymNode Eq(u0, 1)).
            self._graph_lower_disabled = bool(state_dict[key])

        # Back-compat: checkpoints predating the version buffer were written
        # under version 1.1, the last one released before it existed.
        version_key = prefix + "version_tensor"
        if version_key not in state_dict:
            state_dict[version_key] = self.version_tensor.new_tensor(1.1)
        state_dict[version_key] = self.version_tensor.new_tensor(
            self._migrate_variables(
                state_dict, float(state_dict[version_key].item()), prefix
            )
        )

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self.version = float(self.version_tensor.item())

    def _run_tilelang_edge_geometry(
        self, edge_vec: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            raise RuntimeError("DP_TILELANG_EDGE_GEOMETRY is inference-only")
        if (
            edge_vec.dtype != torch.float32
            or not edge_vec.is_cuda
            or torch.version.hip is None
        ):
            raise RuntimeError(
                "TileLang edge geometry requires float32 HIP/DCU tensors"
            )
        from kernels.tilelang_ops import tilelang_edge_geometry

        return tilelang_edge_geometry(edge_vec, eps)

    def _run_tilelang_edge_quaternion(
        self, edge_vec: torch.Tensor, _edge_len: torch.Tensor, eps: float
    ) -> torch.Tensor:
        if self.training:
            raise RuntimeError("DP_TILELANG_EDGE_QUATERNION is inference-only")
        if (
            edge_vec.dtype != torch.float32
            or not edge_vec.is_cuda
            or torch.version.hip is None
        ):
            raise RuntimeError(
                "TileLang edge quaternion requires float32 HIP/DCU tensors"
            )
        from kernels.tilelang_ops import tilelang_edge_quaternion

        return tilelang_edge_quaternion(edge_vec, eps, edge_len=_edge_len)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._tilelang_dpa4_enabled:
            if self.training:
                raise RuntimeError("DP_TILELANG_DPA4 is inference-only")
            if (
                self.lmax != 1
                or self.mmax != 1
                or self.n_focus != 1
                or self.n_atten_head != 1
                or self.radial_so2_mode != "none"
                or self.edge_cartesian
                or self.bridging_switch is not None
                or self.spin_embedding is not None
                or self.charge_spin_embedding is not None
                or self.so3_readout != "none"
                or self.use_full_attn_res
                or self.use_block_attn_res
                or self.ffn_so3_grid
                or any(self.grid_mlp)
                or any(self.grid_branch)
                or self.node_wise_s2
                or self.node_wise_so3
                or self.message_node_s2
                or self.message_node_so3
                or self.atten_f_mix
                or self.use_atten_v_proj
                or self.use_atten_o_proj
                or any(
                    block.so2_conv.node_wise_grid_product is not None
                    or block.so2_conv.message_node_grid_product is not None
                    or block.so2_conv.node_cartesian_tp is not None
                    for block in self.blocks
                )
            ):
                raise RuntimeError(
                    "DP_TILELANG_DPA4 currently supports the single-focus, lmax=mmax=1 "
                    "water inference configuration without spin, bridging or SO(3) readout"
                )
        return self.call(*args, **kwargs)

    def _forward_blocks(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """Run the interaction blocks under the pt_expt AMP policy.

        This is the torch (pt_expt) implementation of the descriptor's
        ``use_amp`` switch, mirroring the reference pt ``_compute_mode_ctx``:
        bfloat16 autocast wraps only the interaction-block region, while the
        geometry, edge cache, radial, env-seed, GIE and output FFN stages stay
        in fp32 (or higher). The dpmodel base stores ``use_amp`` only as a
        config flag and never autocasts (array-API has no autocast), so the
        real automatic mixed precision lives here.

        Training follows ``use_amp`` and evaluation follows ``DP_AMP_INFER``
        (captured once at construction as ``use_amp_infer``). The two are
        independent: mixed precision at inference is a throughput choice that
        must not require a model to have been trained with it. ``x`` is the
        node-feature tensor entering the blocks, and its device is the working
        device.
        """
        enabled = self.use_amp if self.training else self.use_amp_infer
        if enabled and x.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                return super()._forward_blocks(x, *args, **kwargs)
        return super()._forward_blocks(x, *args, **kwargs)

    def share_params(
        self,
        base_class: "DescrptDPA4",
        shared_level: int,
        model_prob: float = 1.0,
        resume: bool = False,
    ) -> None:
        # Multi-task parameter sharing for DPA4 is out of scope for this PR.
        raise NotImplementedError("share_params is not yet implemented for DescrptDPA4")
