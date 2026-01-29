
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LoRALayer 
from typing import Optional, List
import os
import csv
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import json
import numpy as np
import time
import re
class SVDLinear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            self.lora_E = nn.Parameter(
                self.weight.new_zeros(r, 1)
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear 
            # and E (singular values) for zero 
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(
                    self.lora_B @ (self.lora_A*self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(
                    self.lora_B @ (self.lora_A * self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                try:
                    result += (
                        self.lora_dropout(x) @ (self.lora_A * (self.lora_E)).T @ self.lora_B.T
                    ) * self.scaling / (self.ranknum+1e-5)
                except Exception as e:
                    print(e)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class RankAllocator(object):
    def __init__(
        self, model, 
        init_warmup:int, 
        final_warmup:int,
        mask_interval:int,
        total_step:Optional[int]=None,
        tb_writter=None, 
        tb_writter_loginterval=500, k=2, b=4,
        output_dir=None, 
        enable_scheduler=False
    ):
        self.k = k
        self.b = b
        self.initial_b = b
        self.enable_scheduler = enable_scheduler
        self.output_dir = output_dir
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup 
        self.mask_interval = mask_interval
        self.total_step = total_step
        self.model = model
        self.rank_pattern = {} 
        self.get_lora_param_name()
        
        def extract_layer_number(name):
            match = re.search(r'\.layer\.(\d+)\.', name)
            return int(match.group(1)) if match else float('inf')

        self.rank_names = sorted(
            [n for n, _ in model.named_parameters() if "lora_E" in n],
            key=extract_layer_number
        )

        
        # self._csv_header_written = os.path.exists(self.csv_path)
        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval 

    def get_lora_param_name(self):
        self.name_set = set() 
        self.total_rank = 0 
        self.shape_dict = {}
        for n,p in self.model.named_parameters():
            if "lora_A" in n: 
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0) 
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set)) 
        

    def compute_matrix_importance(self, name: str, E: torch.Tensor) -> float:
        with torch.no_grad():
            p = E ** 2
            p = p / p.sum()
            rank = E.numel()
            entropy = -torch.sum(p * torch.log(p + 1e-8))
            entropy = entropy / math.log(rank)
            return entropy.item()
        
            

    def mask_to_target_rank(self, model, curr_rank):
        lora_A_list = []
        lora_B_list = []
        lora_E_list = []
        lora_E_name_map = {}

        for n, p in model.named_parameters(): 
            if "lora_A" in n: lora_A_list.append(p)
            if "lora_B" in n: lora_B_list.append(p)
            if "lora_E" in n: 
                lora_E_list.append(p)
                lora_E_name_map[p] = n

        importance_matrix_level_all = []  
        importance_matrix_level_r_gt_1 = []  
        valid_idx_r_gt_1 = [] 
        valid_idx_all = []  

        for idx, (A, B, E) in enumerate(zip(lora_A_list, lora_B_list, lora_E_list)):
            name = lora_E_name_map[E]
            importance = self.compute_matrix_importance(name, E)
            importance_matrix_level_all.append(importance)
            valid_idx_all.append(idx)

            r = E.shape[0]
            if r > 1:
                importance_matrix_level_r_gt_1.append(importance)
                valid_idx_r_gt_1.append(idx)

        # ===== Decrease =====
        importance_tensor_decrease = torch.tensor(importance_matrix_level_r_gt_1)
        decrease_idx = torch.topk(importance_tensor_decrease, self.b, largest=False).indices.tolist()
        decrease_idx = [valid_idx_r_gt_1[i] for i in decrease_idx]

        # ===== Increase =====
        importance_tensor_increase = torch.tensor(importance_matrix_level_all)
        increase_idx = torch.topk(importance_tensor_increase, self.b, largest=True).indices.tolist()
        increase_idx = [valid_idx_all[i] for i in increase_idx]
        # === Decrease rank ===
        for i in decrease_idx:
            A = lora_A_list[i]  # shape: (r, in_dim)
            B = lora_B_list[i]  # shape: (out_dim, r)
            E = lora_E_list[i]  # shape: (r, 1)

            r = E.shape[0]
            if r <= 1:
                continue  

            min_energy_idx = int(torch.argmin(E))

            keep_indices = [j for j in range(r) if j != min_energy_idx]
            keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=A.device)

            A_new = torch.nn.Parameter(A[keep_indices])
            B_new = torch.nn.Parameter(B[:, keep_indices])
            E_new = torch.nn.Parameter(E[keep_indices])

            lora_A_list[i] = A_new
            lora_B_list[i] = B_new
            lora_E_list[i] = E_new

            self._replace_param(model, A, A_new)
            self._replace_param(model, B, B_new)
            self._replace_param(model, E, E_new)
        
        # === Increase rank ===
        for i in increase_idx:
            A, B, E = lora_A_list[i], lora_B_list[i], lora_E_list[i]
            hdim_a = A.shape[1]
            hdim_b = B.shape[0]
            new_a = torch.randn(1, hdim_a, device=A.device)
            new_b = torch.randn(hdim_b, 1, device=B.device)
            new_e = torch.zeros_like(E[0:1])  
            A_new = torch.nn.Parameter(torch.cat([A, new_a], dim=0))
            B_new = torch.nn.Parameter(torch.cat([B, new_b], dim=1))
            E_new = torch.nn.Parameter(torch.cat([E, new_e], dim=0))
            lora_A_list[i] = A_new
            lora_B_list[i] = B_new
            lora_E_list[i] = E_new
            self._replace_param(model, A, A_new)
            self._replace_param(model, B, B_new)
            self._replace_param(model, E, E_new)

        for name in self.rank_names:
            param = dict(model.named_parameters())[name]
            self.rank_pattern[name] = param.size(0)
        
        # if not self._csv_header_written:
        #     with open(self.csv_path, "w", newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerow(["step"] + self.rank_names)
        #     self._csv_header_written = True

        # row = [self.global_step] + [self.rank_pattern[name] for name in self.rank_names]
        # with open(self.csv_path, "a", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(row)

        # if not os.path.exists(self.csv_path):
        #     print("[plot_rank_heatmap] rank_log.csv not exist")
        #     return True

        # df = pd.read_csv(self.csv_path)
        # if df.shape[0] < 2:
        #     return True

        # df = df.tail(8)

        # if "step" not in df.columns:
        #     print("[plot_rank_heatmap] step not in df.columns")
        #     return
        # df.set_index("step", inplace=True)

        # plt.figure(figsize=(12, max(4, len(df.columns) // 2)))
        # ax = sns.heatmap(
        #     df.T,
        #     cmap="YlGnBu",
        #     annot=True,           
        #     fmt=".0f",           
        #     linewidths=0.5,
        #     linecolor='gray',
        #     cbar_kws={'label': 'Rank Size'}
        # )
        # plt.title("LoRA Rank Heatmap (Last 8 Steps)")
        # plt.xlabel("Training Step")
        # plt.ylabel("LoRA Modules")
        # plt.tight_layout()

        # step = df.index[-1]  
        # filename = f"rank_heatmap_step_{step}.png"
        # save_path = os.path.join(self.output_dir if self.output_dir else ".", filename)
        # plt.savefig(save_path)
        # plt.close()
        # print(f"[plot_rank_heatmap] Rank heatmap saved to {save_path}")
        
        return True

    def _replace_param(self, model, old_param, new_param):
        for module_name, module in model.named_modules():
            for name, param in module.named_parameters(recurse=False):
                if param is old_param:
                    # Register properly
                    setattr(module, name, new_param)
                    return

    def update_and_mask(self, model, global_step):
        self.global_step = global_step
        if global_step < self.total_step - self.final_warmup:
            
            if self.enable_scheduler:
                # print("[Scheduler] Now is enabled")
                self._b_scheduler(global_step)
            if global_step > self.initial_warmup and (global_step - self.initial_warmup) % self.mask_interval == 0 and self.b > 0:
                print(f"[Masking] Step={global_step}, b={self.b}")
                return 0, self.mask_to_target_rank(model, 0)
        # if self.global_step== self.total_step-self.final_warmup :
        #     df = pd.read_csv(self.csv_path)
        #     if "step" not in df.columns:
        #         print("[plot_rank_heatmap] "step" not in df.columns")
        #         return 0, None
        #     df.set_index("step", inplace=True)
        #     plt.figure(figsize=(12, max(4, len(df.columns) // 2)))
        #     ax = sns.heatmap(
        #         df.T,
        #         cmap="YlGnBu",
        #         annot=False,
        #         linewidths=0.5,
        #         linecolor='gray',
        #         cbar_kws={'label': 'Rank Size'}
        #     )

        #     plt.title("LoRA Rank Heatmap (Final)")
        #     plt.xlabel("Training Step")
        #     plt.ylabel("LoRA Modules")
        #     plt.tight_layout()

        #     step = df.index[-1]  
        #     filename = f"rank_heatmap_final.png"
        #     save_path = os.path.join(self.output_dir if self.output_dir else ".", filename)
        #     plt.savefig(save_path)
        #     plt.close()
        #     print(f"[plot_rank_heatmap] Rank heatmap saved to {save_path}")
        return 0, None

    def _b_scheduler(self, global_step):
        initial_b = self.initial_b
        final_b = 0
        total_step = self.total_step
        progress = (global_step - self.initial_warmup) / (total_step - self.final_warmup - self.initial_warmup)
        progress = min(max(progress, 0), 1)
        mul_coeff = progress ** 3
        self.b = round(initial_b + (final_b - initial_b) * mul_coeff)

    def get_rank_pattern(self):
        return self.rank_pattern

    def set_total_step(self, total_step):
        self.total_step = total_step
        assert self.total_step > self.initial_warmup + self.final_warmup





def compute_orth_regu(model, regu_weight=0.1):
    # The function to compute orthongonal regularization for SVDLinear in model. 
    regu_loss, num_param = 0., 0
    for n,p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p 
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov-I, p="fro")
            num_param += 1
    return regu_weight*regu_loss/num_param