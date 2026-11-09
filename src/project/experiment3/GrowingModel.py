import copy

import torch
import torch.nn as nn

from project.experiment3.BaselineModel import _Model2DArchitecture, _Model3DArchitecture


class _GrowthMixin:
    proportions = [0.1, 0.5, 1.0]

    def _initialize_growth(self) -> None:
        self.proportions = self.proportions.copy()
        self.growth_index = -1
        self.growth_proportion = 0.0
        self.prop_it = iter(enumerate(self.proportions))
        self.grow()

    def grow(self) -> int:
        try:
            index, proportion = next(self.prop_it)
        except StopIteration:
            print("All Grown Up!")
            return -1

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                if index == 0:
                    module.old_size = 0
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    module.old_size = module.cur_size
                module.cur_size = int(module.weight.shape[0] * proportion)
                module.grad_mask = torch.zeros_like(module.weight, device=self.device)
                module.grad_mask[:module.cur_size] = 1
                nn.init.kaiming_normal_(
                    module.weight[module.old_size:module.cur_size],
                    mode="fan_out",
                    nonlinearity="relu",
                )
                nn.init.zeros_(module.weight[module.cur_size:])
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.growth_index = index
        self.growth_proportion = proportion
        return index

    @property
    def growth_complete(self) -> bool:
        return self.growth_index == len(self.proportions) - 1

    def get_growth_metadata(self) -> dict[str, int | float | list[float]]:
        return {
            "growth_index": self.growth_index,
            "growth_proportion": self.growth_proportion,
            "proportions": self.proportions.copy(),
        }

    def restore_growth_metadata(self, metadata: dict[str, int | float | list[float]]) -> None:
        proportions = [float(value) for value in metadata["proportions"]]
        growth_index = int(metadata["growth_index"])
        growth_proportion = float(metadata["growth_proportion"])
        if not 0 <= growth_index < len(proportions):
            raise ValueError(f"Invalid growth index: {growth_index}")
        if proportions[growth_index] != growth_proportion:
            raise ValueError("Growth proportion does not match the saved growth index")

        self.proportions = proportions
        self.growth_index = growth_index
        self.growth_proportion = growth_proportion
        self.prop_it = iter(enumerate(proportions[growth_index + 1:], start=growth_index + 1))
        previous_proportion = proportions[growth_index - 1] if growth_index else 0.0
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                module.old_size = int(module.weight.shape[0] * previous_proportion)
                module.cur_size = int(module.weight.shape[0] * growth_proportion)
                module.grad_mask = torch.zeros_like(module.weight, device=self.device)
                module.grad_mask[:module.cur_size] = 1

    def to_compact_model(self):
        """Return a copy containing only channels and units active at this phase."""
        compact = copy.deepcopy(self)
        active_channels = self.features[0].in_channels
        for index, module in enumerate(self.features):
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                output_channels = module.cur_size
                compact_module = copy.deepcopy(module)
                compact_module.in_channels = active_channels
                compact_module.out_channels = output_channels
                compact_module.weight = nn.Parameter(
                    module.weight[:output_channels, :active_channels].detach().clone(),
                    requires_grad=module.weight.requires_grad,
                )
                if module.bias is not None:
                    compact_module.bias = nn.Parameter(
                        module.bias[:output_channels].detach().clone(),
                        requires_grad=module.bias.requires_grad,
                    )
                compact_module.cur_size = output_channels
                for attribute in ("old_size", "grad_mask"):
                    if hasattr(compact_module, attribute):
                        delattr(compact_module, attribute)
                compact.features[index] = compact_module
                active_channels = output_channels
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                compact_module = copy.deepcopy(module)
                compact_module.num_features = active_channels
                if module.affine:
                    compact_module.weight = nn.Parameter(
                        module.weight[:active_channels].detach().clone(),
                        requires_grad=module.weight.requires_grad,
                    )
                    compact_module.bias = nn.Parameter(
                        module.bias[:active_channels].detach().clone(),
                        requires_grad=module.bias.requires_grad,
                    )
                if module.track_running_stats:
                    compact_module.running_mean = module.running_mean[:active_channels].detach().clone()
                    compact_module.running_var = module.running_var[:active_channels].detach().clone()
                compact.features[index] = compact_module

        active_features = active_channels * (self.fc[0].in_features // self.features[8].out_channels)
        for index, module in enumerate(self.fc):
            if isinstance(module, nn.Linear):
                output_features = module.cur_size
                compact_module = copy.deepcopy(module)
                compact_module.in_features = active_features
                compact_module.out_features = output_features
                compact_module.weight = nn.Parameter(
                    module.weight[:output_features, :active_features].detach().clone(),
                    requires_grad=module.weight.requires_grad,
                )
                if module.bias is not None:
                    compact_module.bias = nn.Parameter(
                        module.bias[:output_features].detach().clone(),
                        requires_grad=module.bias.requires_grad,
                    )
                compact_module.cur_size = output_features
                for attribute in ("old_size", "grad_mask"):
                    if hasattr(compact_module, attribute):
                        delattr(compact_module, attribute)
                compact.fc[index] = compact_module
                active_features = output_features
            elif isinstance(module, nn.BatchNorm1d):
                compact_module = copy.deepcopy(module)
                compact_module.num_features = active_features
                if module.affine:
                    compact_module.weight = nn.Parameter(
                        module.weight[:active_features].detach().clone(),
                        requires_grad=module.weight.requires_grad,
                    )
                    compact_module.bias = nn.Parameter(
                        module.bias[:active_features].detach().clone(),
                        requires_grad=module.bias.requires_grad,
                    )
                if module.track_running_stats:
                    compact_module.running_mean = module.running_mean[:active_features].detach().clone()
                    compact_module.running_var = module.running_var[:active_features].detach().clone()
                compact.fc[index] = compact_module
        return compact


class GrowingModel2D(_GrowthMixin, _Model2DArchitecture):
    model_name = "GrowingModel"

    def __init__(self, num_channels=3, size=28, name=None, device="cuda"):
        super().__init__(num_channels=num_channels, size=size, name=name, device=device)
        self._initialize_growth()


class GrowingModel3D(_GrowthMixin, _Model3DArchitecture):
    model_name = "GrowingModel"

    def __init__(self, num_channels=1, size=28, name=None, device="cuda"):
        super().__init__(num_channels=num_channels, size=size, name=name, device=device)
        self._initialize_growth()


class GrowingModel:
    def __new__(
        cls,
        num_channels: int = 3,
        size: int = 28,
        name: str | None = None,
        device: str | torch.device = "cuda",
        spatial_dims: int = 2,
    ) -> nn.Module:
        if spatial_dims == 2:
            model_type = GrowingModel2D
        elif spatial_dims == 3:
            model_type = GrowingModel3D
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        return model_type(num_channels=num_channels, size=size, name=name, device=device)