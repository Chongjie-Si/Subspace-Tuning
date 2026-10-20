import torch

from loralib.layertsd import LayerTSDLinear


class BudgetController:
    """Measure LayerTSD sensitivity and allocate a global direction budget."""

    def calculate_sensitivities(self, model):
        sensitivities = {}
        with torch.no_grad():
            for name, layer in model.named_modules():
                if not isinstance(layer, LayerTSDLinear):
                    continue
                if not layer._svd_cached:
                    layer.cache_svd()

                u = layer.svd_u.detach()
                sigma = layer.svd_sigma.detach()
                vh = layer.svd_vh.detach()
                delta_w = layer.get_delta_w().detach()
                if u.ndim != 2 or vh.ndim != 2 or sigma.ndim != 1:
                    raise ValueError(f"Invalid cached SVD shapes for {name}")
                if u.shape[0] != delta_w.shape[0] or vh.shape[1] != delta_w.shape[1]:
                    raise ValueError(f"SVD cache does not match delta_w for {name}")

                # U.T @ Delta_W @ V projects the update onto base singular
                # directions; the diagonal is Delta_sigma_i.
                projected_change = torch.diagonal(u.T @ delta_w @ vh.T)
                change_rates = projected_change.abs() / (sigma[: projected_change.numel()].abs() + 1e-6)
                top_k = min(max(layer.r, 1), change_rates.numel())
                sensitivities[name] = float(torch.topk(change_rates, top_k).values.sum().item())
        return sensitivities

    def allocate_and_initialize(self, model, total_budget: int = 192):
        if not isinstance(total_budget, int) or total_budget < 1:
            raise ValueError(f"total_budget must be a positive integer, got {total_budget!r}")

        sensitivities = self.calculate_sensitivities(model)
        if not sensitivities:
            self.last_allocations = {}
            return []
        total_sensitivity = sum(sensitivities.values())
        if total_sensitivity <= 0.0:
            equal_share = max(1, round(total_budget / len(sensitivities)))
            allocations = {name: equal_share for name in sensitivities}
        else:
            allocations = {
                name: max(1, round(total_budget * sensitivity / total_sensitivity))
                for name, sensitivity in sensitivities.items()
            }

        parameters = []
        for name, layer in model.named_modules():
            if isinstance(layer, LayerTSDLinear) and name in allocations:
                parameters.append(layer.initialize_dash(allocations[name]))
        self.last_allocations = allocations
        return parameters