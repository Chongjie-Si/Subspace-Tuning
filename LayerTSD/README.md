# LayerTSD

LayerTSD (**Layer**-wise **T**ask-**S**pecific **D**irections) is a subspace-finetuning method built on top of LoRA. Each injected LoRA-style layer gets a frozen base weight, a standard low-rank LoRA path (Lora A/B), and starting at training step 100 a trainable per-layer vector of direction scales. These scales are allocated across the layers by a budget controller: the budget is split proportional to each layer's measured sensitivity along the base weight's singular directions.

The result is a single extra optimizer param-group whose dimensionality is decided *during* training, so no per-layer rank needs to be chosen up front.

## Files

| File | Role |
| --- | --- |
| `loralib/loralib/layertsd.py` | `LayerTSDLinear` layer (LoRA + SVD direction cache + `dash_directions` parameter) |
| `budget_controller.py` | `BudgetController` — sensitivity measurement and budget allocation |
| `NLU/src/transformers/models/deberta_v2/modeling_deberta_v2.py` | Injection points (6 `lora_type == "svd"` branches) |
| `NLU/src/transformers/models/deberta_v2/configuration_deberta_v2.py` | `apply_lora`, `lora_type`, `lora_module`, `lora_r`, `lora_alpha` config fields |
| `NLU/src/transformers/trainer.py` | Step-100 allocation hook + scheduler fix |
| `NLU/examples/text-classification/run_glue.py` | Wires LayerTSD through `AutoConfig` for GLUE tasks |

## How it works

### 1. `LayerTSDLinear` (`layertsd.py`)

`class LayerTSDLinear(nn.Linear, LoRALayer)` — one forward pass composes three terms:

```
y = W x                       base frozen weight
  + dropout(x) @ A^T @ B^T    standard LoRA path (scaled by lora_alpha / r)
  + dropout(x) @ update^T     SVD direction update, active only after allocation
```

- **LoRA path.** `lora_A` (`r x in`), `lora_B` (`out x r`), initialized Kaiming-uniform / zeros, `scaling = lora_alpha / r`. Base `weight` is frozen (`requires_grad = False`).
- **SVD cache.** On first forward, `cache_svd()` computes `torch.linalg.svd(weight, full_matrices=False)` and stores `svd_u`, `svd_sigma`, `svd_vh` as non-persistent buffers.
- **Direction parameter.** `initialize_dash(s_l)` creates a trainable `dash_directions` parameter of length `s_l`. Once set, the forward adds:

```
U[:, :k] @ diag(dash_directions[:k]) @ Vh[:k, :]        where k = min(s_l, rank(weight))
```

so the trainable scales modulate the top `k` singular directions of the base weight.
- `get_delta_w()` returns the current low-rank LoRA update (`B @ A`, scaled) for sensitivity analysis.

### 2. `BudgetController` (`budget_controller.py`)

**`calculate_sensitivities(model)`** — for every `LayerTSDLinear` in the model:

1. Project the LoRA update onto the base SVD basis: `diag(U^T @ delta_w @ Vh^T)` gives per-direction change `Δσ_i`.
2. `change_rate_i = |Δσ_i| / (σ_i + 1e-6)` — normalized change per base singular direction.
3. Sensitivity = sum of the top-`max(r, 1)` change rates.

Raises `ValueError` if cached SVD shapes don't match `delta_w`.

**`allocate_and_initialize(model, total_budget=192)`**

1. Computes sensitivities.
2. If total sensitivity is 0 (or no layers), falls back to an equal per-layer share.
3. Otherwise allocates budget proportionally:

```
s_l = max(1, round(total_budget * sensitivity / total_sensitivity))
```

4. Calls `initialize_dash(s_l)` on each allocated layer.
5. Stores the allocation map in `last_allocations` and returns the new parameters.

### 3. Trainer hook (`trainer.py`)

The step-100 hook sits immediately before `self.state.global_step += 1` (line 1221):

- On `global_step == 100`, creates `BudgetController()` and calls `allocate_and_initialize(self.model, total_budget=192)`.
- If new parameters were created, adds them to the optimizer with `add_param_group(...)`.
- **Scheduler fix:** after `add_param_group`, both `lr_scheduler.base_lrs` and `lr_scheduler.lr_lambdas` are extended with one copy of the first value, so the next `lr_scheduler.step()` doesn't crash with `zip() argument 2 is shorter`.
- Prints `LayerTSD allocation: <json>` and writes `results/allocations_{base|large}_{task}.json` (model type determined from `num_hidden_layers >= 24`).

Imports added: `import json` (line 22) and `from budget_controller import BudgetController` (line 34).

## Injection points

All six injection sites in `modeling_deberta_v2.py` use the form:

```python
elif config.lora_type == "svd":
    self.<proj> = lora.LayerTSDLinear(<in>, <out>, r=config.lora_r,
                                      lora_alpha=config.lora_alpha, merge_weights=False)
```

| Location | Layer name | Line |
| --- | --- | --- |
| `DebertaV2SelfOutput` | `attention.output` dense | 232 |
| `DebertaV2Intermediate` | `intermediate` dense | 298 |
| `DebertaV2Output` | `layer.output` dense | 324 |
| `DisentangledSelfAttention` | `query_proj` | 622 |
| `DisentangledSelfAttention` | `key_proj` | 634 |
| `DisentangledSelfAttention` | `value_proj` | 646 |

Module is imported as `import loralib as lora` (line 17). For `DebertaV2Model` with `L` layers and all six modules enabled, 6·L `LayerTSDLinear` layers are injected.

### Config fields (`configuration_deberta_v2.py`)

| Field | Default | Meaning |
| --- | --- | --- |
| `apply_lora` | `False` | globally enable the LoRA-style branches |
| `lora_type` | `"frd"` | `"svd"` selects `LayerTSDLinear` |
| `lora_module` | `"query,value"` | comma list: `query,key,value,intermediate,layer.output,attention.output` |
| `lora_r` | `None` | LoRA rank |
| `lora_alpha` | `None` | LoRA alpha |

### CLI (`run_glue.py`)

`run_glue.py` forwards `apply_lora`, `lora_type`, `lora_module`, `lora_alpha`, `lora_r` (lines 433–437) verbatim into `AutoConfig.from_pretrained`, so a GLUE run needs no code changes:

```bash
python run_glue.py \
  --model_name_or_path microsoft/deberta-v2-xlarge \
  --task_name cola \
  --apply_lora \
  --lora_type svd \
  --lora_module "query,key,value,attention.output,intermediate,layer.output" \
  --lora_r 8 \
  --lora_alpha 16 \
  --do_train --do_eval
```

The base weight SVD is cached lazily on first forward; the `dash_directions` param-group is only created at step 100.

## Implementation

1. **`loralib/loralib/layertsd.py`** — `LayerTSDLinear` with lazy `dash_directions` initialization (`initialize_dash`), detached SVD caching (`cache_svd`), and `get_delta_w()`.
2. **`loralib/loralib/__init__.py`** — export the layer: `from .layertsd import *`.
3. **`budget_controller.py`** — sensitivity calculation (`calculate_sensitivities`) and dynamic budget allocation (`allocate_and_initialize`, `total_budget=192`).
4. **Inject the layer into the target model.** For each linear to adapt, add an `lora_type == "svd"` branch that constructs `lora.LayerTSDLinear(<in>, <out>, r=config.lora_r, lora_alpha=config.lora_alpha, merge_weights=False)` (mirror `modeling_deberta_v2.py` lines 232/298/324/622/634/646).
5. **Wire the step-100 hook into `trainer.py`.** Add two imports at the top:

```python
import json
from budget_controller import BudgetController
```

Then insert the block below before `self.state.global_step += 1` (after the optimizer/scheduler handling). The scheduler extension is **required** — without it the newly added optimizer param-group crashes the next `lr_scheduler.step()` with `zip(): argument 2 is shorter`:

```python
if self.state.global_step == 100:
    budget_controller = BudgetController()
    new_parameters = budget_controller.allocate_and_initialize(
        self.model,
        total_budget=192,
    )

    if new_parameters:
        self.optimizer.add_param_group({"params": new_parameters})

        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, "base_lrs"):
            base_lrs = list(self.lr_scheduler.base_lrs)
            base_lrs.append(base_lrs[0])
            self.lr_scheduler.base_lrs = base_lrs
            if hasattr(self.lr_scheduler, "lr_lambdas") and self.lr_scheduler.lr_lambdas:
                lr_lambdas = list(self.lr_scheduler.lr_lambdas)
                lr_lambdas.append(lr_lambdas[0])
                self.lr_scheduler.lr_lambdas = lr_lambdas

    print(
        "LayerTSD allocation:",
        json.dumps(budget_controller.last_allocations, sort_keys=True),
    )
```

6. **Expose the config knobs.** Add `apply_lora`, `lora_type`, `lora_module`, `lora_r`, `lora_alpha` to the model config and forward them through `AutoConfig` (as `run_glue.py` lines 433–437 do).

## Verification

A 105-step Trainer harness run (real `Trainer`, `BudgetController`, `LayerTSDLinear`, linear scheduler with warmup) produced:

- All 12 `LayerTSDLinear` layers budget-allocated at step 100.
- Training completed to `global_step=105` with no scheduler crash.
- Post-training: `lr_lambdas: 2`, `base_lrs: 2` — the extended-group fix holds.
- Example allocation at step 100: `{"attn_out": 32, "inter": 1, "k": 31, "out": 1, "q": 77, "v": 51}`.

A standalone functional check of the model + controller in this environment:

```
LayerTSDLinear count: 12
FORWARD_BACKWARD_OK, loss: ...
allocated params: 12
POST_ALLOC_FORWARD_OK
FUNCTIONAL_TEST_PASS
```

## Known limitations

- Only DeBERTa-v2 (encoder-only GLUE path) is wired up. Decoder-only models and VLMs need the same `lora_type == "svd"` injection added to their modeling files; `LayerTSDLinear` and `BudgetController` themselves are architecture-agnostic.
- `cache_svd()` computes a full SVD of the base weight (O(n³)); very wide decoder layers (>~8k) would benefit from a truncated/top-k SVD.
- `total_budget=192` is hardcoded in the trainer hook, so larger models simply get smaller per-layer allocations unless the hook is parameterized.
- The bundled binary is a 2021-era transformers fork (v4.4.2); it does not run end-to-end on the Python 3.14 environment used for verification (legacy pip version-specifier parsing, `datasets`/`dill` pickling, hub access) — the harness and functional test bypass those issues. Model/trainer correctness is validated independently of that.