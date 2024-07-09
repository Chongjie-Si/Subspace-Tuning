#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
from typing import Optional, List

def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
        src_weight=None
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

        #self.dora_init(src_weight)
    

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
                           merge_weights=merge_weights, src_weight=self.weight)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.norm_init = torch.linalg.norm(self.weight, dim=1).view(-1,1).to(self.lora_A.device)
            self.weigh_m_wdecomp = nn.Parameter(self.norm_init)
            self.weigh_m_wdecomp.requires_grad = True
            self.new_lora = nn.Parameter(torch.zeros(in_features, 1))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    
    def cal(self, weight1, lora1):
        return weight1.T

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            if torch.linalg.norm(self.lora_B) == 0:
                self.weigh_m_wdecomp = nn.Parameter(torch.linalg.norm(self.weight, dim=1).view(-1,1))
                self.weigh_m_wdecomp.requires_grad = True
            new_weight = self.weight + (self.lora_B @ self.lora_A) * self.scaling
            norm_scale = self.weigh_m_wdecomp.view(-1) / torch.linalg.norm(new_weight, dim=1).detach()
            org_result = F.linear(x, T(self.weight))
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale - 1) * F.linear(dropout_x, T(self.weight))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)
            result += norm_scale * (dropout_x @ self.lora_A.T @ self.lora_B.T)* self.scaling
            return result
        else:
            return F.linear(x, T(self.weight+self.new_lora.view(-1)), bias=self.bias)

