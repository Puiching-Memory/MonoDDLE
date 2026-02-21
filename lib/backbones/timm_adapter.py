import torch
import torch.nn as nn

class TimmBackboneAdapter(nn.Module):
    """Adapter that exposes timm backbone features for CenterNet3D."""

    def __init__(self, model_name, pretrained=True, feature_strides=(4, 8, 16, 32), freeze=False):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError('timm is required for timm backbone support. Please install it with `pip install timm`.') from e

        if not model_name:
            raise ValueError('model_name is required for timm backbone.')

        # First create a temporary model to inspect feature info.
        # Use the per-level 'index' field (model-level index) rather
        # than list position, because some architectures (e.g. DLA)
        # have more internal levels than the default features_only
        # selection exposes, causing an off-by-one when list position
        # is used directly as out_indices.
        temp_model = timm.create_model(model_name, pretrained=False, features_only=True)
        stride_to_index = {fi['reduction']: fi['index'] for fi in temp_model.feature_info}
        del temp_model

        out_indices = []
        for stride in feature_strides:
            if stride in stride_to_index:
                out_indices.append(stride_to_index[stride])
            else:
                raise ValueError(
                    f"Stride {stride} not found in model {model_name} "
                    f"available strides {sorted(stride_to_index.keys())}"
                )

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices
        )
        
        self.channels = self.model.feature_info.channels()
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)
