import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers_lora import LoRALayer


class LayerTSDLinear(nn.Linear, LoRALayer):
    """Linear LoRA adapter whose task-specific directions are allocated later."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        self.fan_in_fan_out = fan_in_fan_out
        self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_empty(r, in_features))
            self.lora_B = nn.Parameter(self.weight.new_empty(out_features, r))
            self.weight.requires_grad = False

        # Its length is decided by BudgetController after the pre-launch phase.
        self.register_parameter("dash_directions", None)
        self.register_buffer("svd_u", torch.empty(0), persistent=False)
        self.register_buffer("svd_sigma", torch.empty(0), persistent=False)
        self.register_buffer("svd_vh", torch.empty(0), persistent=False)
        self._svd_cached = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def cache_svd(self):
        """Cache detached SVD factors of the frozen base weight."""
        weight = self.weight.T if self.fan_in_fan_out else self.weight
        with torch.no_grad():
            # U @ diag(Sigma) @ Vh is the base weight; cache is inference-only.
            u, sigma, vh = torch.linalg.svd(weight.detach(), full_matrices=False)
        self.svd_u = u
        self.svd_sigma = sigma
        self.svd_vh = vh
        self._svd_cached = True

    def initialize_dash(self, s_l: int) -> nn.Parameter:
        """Create the trainable layer-specific direction scales."""
        if not isinstance(s_l, int) or s_l < 1:
            raise ValueError(f"s_l must be a positive integer, got {s_l!r}")
        if self.dash_directions is not None:
            raise RuntimeError("dash directions have already been initialized")
        parameter = nn.Parameter(self.weight.new_zeros(s_l))
        self.dash_directions = parameter
        return parameter

    def get_delta_w(self) -> torch.Tensor:
        """Return the current low-rank update matrix without detaching it."""
        if self.r <= 0:
            return self.weight.new_zeros(self.out_features, self.in_features)
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        return delta_w.T if self.fan_in_fan_out else delta_w

    def forward(self, x: torch.Tensor):
        weight = self.weight.T if self.fan_in_fan_out else self.weight
        if self.r <= 0 or self.merged:
            return F.linear(x, weight, self.bias)

        if not self._svd_cached:
            self.cache_svd()

        result = F.linear(x, weight, self.bias)
        result = result + (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        if self.dash_directions is not None:
            direction_count = min(
                self.dash_directions.numel(), self.svd_sigma.numel()
            )
            direction_update = self.svd_u[:, :direction_count]
            direction_update = direction_update @ torch.diag(
                self.dash_directions[:direction_count]
            )
            direction_update = direction_update @ self.svd_vh[:direction_count, :]
            result = result + self.lora_dropout(x) @ direction_update.T
        return result