from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch.nn as nn
import torch

if TYPE_CHECKING:
    from .config import DataConfig


@dataclass
class MaxwellModelConfig:
    """Config fields used only by the final explicit Maxwell model."""

    model_name: str = "maxwell"
    output_mode: str = "raw"
    context_size: int = 32
    elastic_hidden_size: int = 64
    viscous_hidden_size: int = 64
    n_visco_modes: int = 4
    step_size: float = 0.005
    integrator: str = "exact"
    tau_min: float = 0.001
    tendon_scale_reference: float = 4.0
    tendon_scale_base_init: float = 1.0
    tendon_scale_exponent_init: float = 0.75
    tendon_scale_exponent_min: float = 0.25

    def build(self, data_config: "DataConfig") -> "MaxwellModel":
        return build_maxwell_model(
            dynamic_features=data_config.dynamic_features,
            static_features=data_config.static_features,
            model_config=self,
        )


# Utility used to initialize parameters that will later pass through softplus.
# This lets us specify a desired positive initial value in ordinary units.
def _inv_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus inverse requires a positive value")
    return math.log(math.expm1(value))


class _OptionalEncoder(nn.Module):
    """Small MLP encoder used for one static-parameter group.

    The final model separates static inputs into elastic, viscous, and protocol
    groups. Each group gets its own encoder so the model can treat them
    differently instead of collapsing everything into one shared embedding.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if in_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Tanh(),
                nn.Linear(out_dim, out_dim),
                nn.Tanh(),
            )
        else:
            self.net = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return torch.zeros(x.shape[0], self.out_dim, device=x.device, dtype=x.dtype)
        return self.net(x)


class MaxwellModel(nn.Module):
    """Coupled Maxwell tendon model with explicit raw tendon scaling.

    High-level structure:
    1. Encode static inputs into separate elastic / viscous / protocol contexts.
    2. Use strain + elastic/protocol context to compute a learned elastic force
       scale F_full(t).
    3. Split that force into an equilibrium fraction and transient Maxwell
       fractions.
    4. Evolve explicit internal Maxwell branch states q_i over time.
    5. Sum equilibrium + transient forces and then scale them with an explicit
       monotone law based on raw tendon stiffness:

           scale(l1) = s0 × (l1 / l_ref)^α

    The key idea is that the model's memory is carried by explicit viscoelastic
    states q_i rather than a generic GRU hidden state.
    """

    def __init__(
        self,
        *,
        dynamic_dim: int,
        static_dim: int,
        elastic_indices: list[int],
        viscous_indices: list[int],
        protocol_indices: list[int],
        context_size: int = 32,
        elastic_hidden_size: int = 64,
        viscous_hidden_size: int = 64,
        n_visco_modes: int = 4,
        step_size: float = 0.005,
        integrator: str = "exact",
        tau_min: float = 1e-3,
        strain_index: int = 0,
        tendon_raw_index: int | None = 0,
        tendon_scale_reference: float = 4.0,
        tendon_scale_base_init: float = 1.0,
        tendon_scale_exponent_init: float = 0.75,
        tendon_scale_exponent_min: float = 0.25,
    ) -> None:
        super().__init__()
        if integrator not in {"euler", "heun", "exact"}:
            raise ValueError(f"Unsupported integrator: {integrator}")
        if n_visco_modes < 1:
            raise ValueError("n_visco_modes must be at least 1")
        if tendon_scale_reference <= 0:
            raise ValueError("tendon_scale_reference must be positive")
        if tendon_scale_base_init <= 0:
            raise ValueError("tendon_scale_base_init must be positive")
        if tendon_scale_exponent_init <= tendon_scale_exponent_min:
            raise ValueError(
                "tendon_scale_exponent_init must exceed tendon_scale_exponent_min"
            )

        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.elastic_indices = list(elastic_indices)
        self.viscous_indices = list(viscous_indices)
        self.protocol_indices = list(protocol_indices)
        self.context_size = context_size
        self.n_visco_modes = n_visco_modes
        self.step_size = float(step_size)
        self.integrator = integrator
        self.tau_min = float(tau_min)
        self.strain_index = strain_index
        self.tendon_raw_index = tendon_raw_index
        self.tendon_scale_reference = float(tendon_scale_reference)
        self.tendon_scale_exponent_min = float(tendon_scale_exponent_min)

        # Separate encoders for the three static-parameter groups.
        # - elastic: tendon stiffness-like parameters
        # - viscous: relaxation magnitude/time parameters
        # - protocol: loading-program parameters such as ramp time
        self.elastic_encoder = _OptionalEncoder(len(self.elastic_indices), context_size)
        self.viscous_encoder = _OptionalEncoder(len(self.viscous_indices), context_size)
        self.protocol_encoder = _OptionalEncoder(
            len(self.protocol_indices), context_size
        )

        # Neural elastic-force law. Given current strain plus elastic/protocol
        # context, it predicts a positive stiffness-like quantity, then force is
        # formed as strain × stiffness. This guarantees zero elastic force at
        # zero strain.
        self.elastic_force_head = nn.Sequential(
            nn.Linear(1 + context_size + context_size, elastic_hidden_size),
            nn.Tanh(),
            nn.Linear(elastic_hidden_size, elastic_hidden_size),
            nn.Tanh(),
            nn.Linear(elastic_hidden_size, 1),
        )

        # Viscous/protocol context is used to determine:
        # - how much force lives in the equilibrium branch vs transient branches
        # - the relaxation times of each transient Maxwell mode
        viscous_static_dim = context_size + context_size
        self.partition_head = nn.Sequential(
            nn.Linear(viscous_static_dim, viscous_hidden_size),
            nn.Tanh(),
            nn.Linear(viscous_hidden_size, n_visco_modes + 1),
        )
        self.tau_head = nn.Sequential(
            nn.Linear(viscous_static_dim, viscous_hidden_size),
            nn.Tanh(),
            nn.Linear(viscous_hidden_size, n_visco_modes),
        )

        # Two learned scalar parameters for the explicit tendon scaling law.
        # They are stored in inverse-softplus form so the actual base/exponent
        # remain positive during training.
        self.tendon_scale_base_raw = nn.Parameter(
            torch.tensor([_inv_softplus(tendon_scale_base_init)], dtype=torch.float32)
        )
        self.tendon_scale_exponent_raw = nn.Parameter(
            torch.tensor(
                [_inv_softplus(tendon_scale_exponent_init - tendon_scale_exponent_min)],
                dtype=torch.float32,
            )
        )
        self.softplus = nn.Softplus()

    def _slice_or_empty(self, static: torch.Tensor, indices: list[int]) -> torch.Tensor:
        if not indices:
            return torch.zeros(
                static.shape[0], 0, device=static.device, dtype=static.dtype
            )
        idx = torch.tensor(indices, device=static.device)
        return torch.index_select(static, dim=1, index=idx)

    def _contexts(
        self, static: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Split normalized static inputs into physical groups, then encode each
        # group separately.
        elastic_x = self._slice_or_empty(static, self.elastic_indices)
        viscous_x = self._slice_or_empty(static, self.viscous_indices)
        protocol_x = self._slice_or_empty(static, self.protocol_indices)
        return (
            self.elastic_encoder(elastic_x),
            self.viscous_encoder(viscous_x),
            self.protocol_encoder(protocol_x),
        )

    def _explicit_tendon_scale(self, static_raw: torch.Tensor | None) -> torch.Tensor:
        # This is the explicit monotone scaling law that was added to improve
        # stiffness extrapolation. It uses raw tendon_l1 rather than a learned
        # embedding, so higher tendon stiffness must increase the force scale.
        if static_raw is None:
            raise ValueError("MaxwellModel requires static_raw inputs")
        batch_size = static_raw.shape[0]
        if (
            self.tendon_raw_index is None
            or self.tendon_raw_index >= static_raw.shape[1]
        ):
            return self.softplus(self.tendon_scale_base_raw).expand(batch_size, 1)
        tendon_l1 = torch.clamp_min(
            static_raw[:, self.tendon_raw_index : self.tendon_raw_index + 1],
            1e-6,
        )
        base = self.softplus(self.tendon_scale_base_raw)
        alpha = self.tendon_scale_exponent_min + self.softplus(
            self.tendon_scale_exponent_raw
        )
        ratio = tendon_l1 / self.tendon_scale_reference
        return base * ratio.pow(alpha)

    def _elastic_force_full(
        self,
        strain: torch.Tensor,
        elastic_ctx: torch.Tensor,
        protocol_ctx: torch.Tensor,
    ) -> torch.Tensor:
        # Learned nonlinear elastic force scale. The softplus keeps the
        # stiffness-like quantity positive, and multiplying by strain enforces
        # F = 0 when strain = 0.
        x = torch.cat([strain, elastic_ctx, protocol_ctx], dim=-1)
        stiffness = self.softplus(self.elastic_force_head(x))
        return strain * stiffness

    def _maxwell_parameters(
        self,
        viscous_ctx: torch.Tensor,
        protocol_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert viscous/protocol context into:
        # - g_eq: equilibrium force fraction
        # - g_modes: transient branch fractions
        # - tau: positive relaxation times for each mode
        x = torch.cat([viscous_ctx, protocol_ctx], dim=-1)
        fractions = torch.softmax(self.partition_head(x), dim=-1)
        g_eq = fractions[:, :1]
        g_modes = fractions[:, 1:]
        tau = self.tau_min + self.softplus(self.tau_head(x))
        return g_eq, g_modes, tau

    def _branch_derivative(
        self,
        q: torch.Tensor,
        drive: torch.Tensor,
        g_modes: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        # Maxwell-style branch dynamics:
        # - branches build in proportion to the positive elastic-force rate
        # - branches relax back toward zero with timescale tau
        return g_modes * drive - q / tau

    def _integrate_branches(
        self,
        q: torch.Tensor,
        drive: torch.Tensor,
        g_modes: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        # Advance the internal Maxwell states one time step.
        #
        # With "exact", each branch uses the closed-form solution of the
        # linear ODE over one step. Euler/Heun are available mainly for
        # experimentation.
        dt = self.step_size
        if self.integrator == "exact":
            decay = torch.exp(-dt / tau)
            q_next = q * decay + (g_modes * drive) * tau * (1.0 - decay)
        else:
            dq1 = self._branch_derivative(q, drive, g_modes, tau)
            if self.integrator == "euler":
                q_next = q + dt * dq1
            else:
                q_euler = q + dt * dq1
                dq2 = self._branch_derivative(q_euler, drive, g_modes, tau)
                q_next = q + 0.5 * dt * (dq1 + dq2)
        return torch.clamp_min(q_next, 0.0)

    def forward(
        self,
        dynamic: torch.Tensor,
        static: torch.Tensor,
        dynamic_raw: torch.Tensor | None = None,
        static_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The constitutive update uses raw physical values (especially strain and
        # raw tendon_l1), so the normalized dynamic tensor is not used here.
        del dynamic
        if dynamic_raw is None:
            raise ValueError("MaxwellModel requires dynamic_raw inputs")

        # Build static contexts and the explicit monotone tendon scaling factor.
        elastic_ctx, viscous_ctx, protocol_ctx = self._contexts(static)
        explicit_scale = self._explicit_tendon_scale(static_raw)
        g_eq, g_modes, tau = self._maxwell_parameters(viscous_ctx, protocol_ctx)

        batch_size, seq_len, _ = dynamic_raw.shape

        # Internal Maxwell branch states q_i. These are the model's explicit
        # viscoelastic memory variables. Starting them at zero ensures F(0)=0
        # when strain is also zero.
        q = torch.zeros(
            batch_size,
            self.n_visco_modes,
            device=dynamic_raw.device,
            dtype=dynamic_raw.dtype,
        )

        # Precompute the learned elastic-force scale at every time step.
        strains = dynamic_raw[:, :, self.strain_index : self.strain_index + 1]
        f_full = []
        for t in range(seq_len):
            f_full.append(
                self._elastic_force_full(strains[:, t, :], elastic_ctx, protocol_ctx)
            )
        f_full = torch.stack(f_full, dim=1)

        preds = []

        # At t=0, transient branch states are zero, so total force is purely the
        # equilibrium part of the elastic force. Because strain(0)=0 in this
        # problem, the full prediction also starts at exactly zero.
        total_force = g_eq * f_full[:, 0, :] + q.sum(dim=-1, keepdim=True)
        preds.append((explicit_scale * total_force).squeeze(-1))

        # March forward in time. The transient branches are driven by the
        # positive rate of change of the elastic-force scale, then decay with
        # their own relaxation times.
        for t in range(seq_len - 1):
            drive = torch.clamp_min(
                (f_full[:, t + 1, :] - f_full[:, t, :]) / self.step_size, 0.0
            )
            q = self._integrate_branches(q, drive, g_modes, tau)
            total_force = g_eq * f_full[:, t + 1, :] + q.sum(dim=-1, keepdim=True)
            preds.append((explicit_scale * total_force).squeeze(-1))

        return torch.stack(preds, dim=1)


def _feature_index(names: list[str], target: str, default: int) -> int:
    try:
        return names.index(target)
    except ValueError:
        return default


def infer_static_feature_groups(
    static_features: list[str],
) -> tuple[list[int], list[int], list[int]]:
    elastic_indices: list[int] = []
    viscous_indices: list[int] = []
    protocol_indices: list[int] = []
    for idx, name in enumerate(static_features):
        if name.startswith("tendon_"):
            elastic_indices.append(idx)
        elif name.startswith("visco_"):
            viscous_indices.append(idx)
        else:
            protocol_indices.append(idx)
    return elastic_indices, viscous_indices, protocol_indices


def build_maxwell_model(
    *,
    dynamic_features: list[str],
    static_features: list[str],
    model_config: MaxwellModelConfig,
) -> MaxwellModel:
    """Build the final explicit Maxwell model from its own config."""
    elastic_indices, viscous_indices, protocol_indices = infer_static_feature_groups(
        static_features
    )
    strain_index = _feature_index(dynamic_features, "strain", 0)
    tendon_raw_index = _feature_index(
        static_features,
        "tendon_l1",
        elastic_indices[0] if elastic_indices else 0,
    )
    return MaxwellModel(
        dynamic_dim=len(dynamic_features),
        static_dim=len(static_features),
        elastic_indices=elastic_indices,
        viscous_indices=viscous_indices,
        protocol_indices=protocol_indices,
        context_size=model_config.context_size,
        elastic_hidden_size=model_config.elastic_hidden_size,
        viscous_hidden_size=model_config.viscous_hidden_size,
        n_visco_modes=model_config.n_visco_modes,
        step_size=model_config.step_size,
        integrator=model_config.integrator,
        tau_min=model_config.tau_min,
        strain_index=strain_index,
        tendon_raw_index=tendon_raw_index if elastic_indices else None,
        tendon_scale_reference=model_config.tendon_scale_reference,
        tendon_scale_base_init=model_config.tendon_scale_base_init,
        tendon_scale_exponent_init=model_config.tendon_scale_exponent_init,
        tendon_scale_exponent_min=model_config.tendon_scale_exponent_min,
    )
