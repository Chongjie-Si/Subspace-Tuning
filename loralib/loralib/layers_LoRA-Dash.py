#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

from .layers_LoRA import LoRALayer

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
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False

            self.index = 8
            self.lora_index = nn.Parameter(self.weight.new_zeros(self.index))
            self.weight_u_top = nn.Parameter(self.weight.new_zeros(out_features, self.index))
            self.weight_vt_top = nn.Parameter(self.weight.new_zeros(self.index, in_features))

            self.warmup = 100
            self.FLAG = 0

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            nn.init.zeros_(self.lora_index)
            nn.init.zeros_(self.weight_u_top)
            nn.init.zeros_(self.weight_vt_top)
    
    def calculate_change_rate(self, a, bb, r):

        self.lora_change_a = nn.Parameter(a)
        self.lora_change_bb = nn.Parameter(bb)

        change_rate = abs(bb) / abs(a)
        _, top_r_indices = torch.topk(change_rate, r)
        return top_r_indices

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

            if self.FLAG < self.warmup:
                if self.FLAG == 0:
                    self.lora_index.requires_grad = False
                    self.weight_u_top.requires_grad = False
                    self.weight_vt_top.requires_grad = False
                self.FLAG += 1
                return result

            elif self.FLAG == self.warmup:
                delta_W = (self.lora_B @ self.lora_A) * self.scaling
                weight_u, weight_sigma, weight_vt = torch.linalg.svd(self.weight, full_matrices=False)
                delta_sigma = torch.diag(torch.matmul(torch.matmul(weight_u.T, delta_W), weight_vt.T))
                top_index = self.calculate_change_rate(weight_sigma, delta_sigma, self.index)

                self.weight_u_top.data = weight_u[:, top_index]
                self.weight_vt_top.data = weight_vt[top_index, :]

                self.lora_index.requires_grad = True
                self.FLAG += 1

            if self.FLAG > self.warmup:
                result += self.lora_dropout(x) @ (self.weight_u_top @ torch.diag(self.lora_index) @ self.weight_vt_top).T
                return result

        else:
            return F.linear(x, T(self.weight), bias=self.bias)

        

