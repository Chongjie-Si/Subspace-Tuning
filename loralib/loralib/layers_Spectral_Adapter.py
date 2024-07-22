#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((out_features, r)))
            self.lora_B = nn.Parameter(torch.zeros((r, in_features)))
            self.scaling = self.lora_alpha / self.r
            self.FLAG = 0
            self.weight.requires_grad = False
            self.reset_parameters()
            # Freezing the pre-trained weight matrix
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            # initialize
            if self.FLAG == 0:
                self.FLAG = 1
                weight_u, weight_sigma, weight_vt = torch.linalg.svd(self.weight, full_matrices=False)
                self.weight_sigma = weight_sigma
                self.lora_A = nn.Parameter(self.weight.new_zeros(self.lora_A.shape))
                self.weight_u = weight_u
                self.weight_vt = weight_vt 
                
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ (
                self.lora_A @ torch.diag(self.weight_sigma[:self.r]) @ self.lora_B + 
                self.weight_u[:, :self.r] @ torch.diag(self.weight_sigma[:self.r]) @ self.lora_B +
                self.lora_A @ torch.diag(self.weight_sigma[:self.r]) @ self.weight_vt[:self.r, :]
                ).T) * self.scaling

            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

