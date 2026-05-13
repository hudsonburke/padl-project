from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .config import DataConfig


@dataclass
class BaselineModelConfig:
    """Config fields used only by the GRU baseline model."""

    model_name: str = "baseline"
    output_mode: str = "normalized"
    hidden_size: int = 64
    context_size: int = 32
    num_layers: int = 2
    dropout: float = 0.1

    def build(self, data_config: "DataConfig") -> "PhysicsAwareRNNBaseline":
        return build_baseline_model(
            dynamic_features=data_config.dynamic_features,
            static_features=data_config.static_features,
            model_config=self,
        )


class PhysicsAwareRNNBaseline(nn.Module):
    """GRU-based baseline used as the main comparison model.

    High-level idea:
    1. Encode the static material/protocol parameters once into a context vector.
    2. Use that context to initialize the GRU hidden state.
    3. Concatenate the same context onto every dynamic time step.
    4. Run the full sequence through a GRU.
    5. Map each GRU hidden state to a force prediction.

    This makes the model history-dependent and parameter-conditioned, but it is
    still a generic sequence model rather than an explicit constitutive solver.
    """

    def __init__(
        self,
        *,
        dynamic_dim: int,
        static_dim: int,
        hidden_size: int = 64,
        context_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        gru_dropout = dropout if num_layers > 1 else 0.0

        # Encode the static inputs (material + protocol parameters) into a
        # learned context vector that summarizes the simulation setting.
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, context_size),
            nn.Tanh(),
            nn.Linear(context_size, context_size),
            nn.Tanh(),
        )

        # Map the context vector into the initial hidden state of the GRU.
        # This lets the recurrent dynamics start from a parameter-dependent
        # state instead of always starting from zeros.
        self.hidden_init = nn.Linear(context_size, num_layers * hidden_size)

        # Sequence model over time. Each step sees the dynamic features
        # (strain, strain rate, phase, ...) plus the repeated static context.
        self.gru = nn.GRU(
            input_size=dynamic_dim + context_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )

        # Readout head that converts each GRU hidden state into a scalar force.
        # The context is concatenated again so the output layer always has
        # direct access to the static parameter information.
        self.readout = nn.Sequential(
            nn.Linear(hidden_size + context_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        dynamic: torch.Tensor,
        static: torch.Tensor,
        dynamic_raw: torch.Tensor | None = None,
        static_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # This baseline uses the normalized dynamic/static inputs prepared by
        # the dataloader. It does not need the raw physical values.
        del dynamic_raw, static_raw

        # Shape: (batch, context_size)
        context = self.static_encoder(static)

        batch_size, seq_len, _ = dynamic.shape

        # Repeat the same static context at every time step so the GRU can use
        # both time-varying inputs and fixed simulation parameters together.
        # Shape: (batch, seq_len, context_size)
        repeated_context = context.unsqueeze(1).expand(
            batch_size, seq_len, context.shape[-1]
        )

        # GRU input per time step = [dynamic features, static context].
        # Shape: (batch, seq_len, dynamic_dim + context_size)
        x = torch.cat([dynamic, repeated_context], dim=-1)

        # Build the initial hidden state from the context and reshape it into
        # the format expected by nn.GRU: (num_layers, batch, hidden_size).
        h0 = (
            self.hidden_init(context)
            .view(self.num_layers, batch_size, self.hidden_size)
            .contiguous()
        )

        # hidden_seq has shape (batch, seq_len, hidden_size).
        hidden_seq, _ = self.gru(x, h0)

        # Predict one scalar force value per time step.
        pred = self.readout(torch.cat([hidden_seq, repeated_context], dim=-1)).squeeze(
            -1
        )
        return pred


def build_baseline_model(
    *,
    dynamic_features: list[str],
    static_features: list[str],
    model_config: BaselineModelConfig,
) -> PhysicsAwareRNNBaseline:
    """Build the GRU baseline from its own model config."""
    return PhysicsAwareRNNBaseline(
        dynamic_dim=len(dynamic_features),
        static_dim=len(static_features),
        hidden_size=model_config.hidden_size,
        context_size=model_config.context_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
    )
