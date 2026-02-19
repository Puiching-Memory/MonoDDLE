import torch
import torch.nn as nn


def _extract_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            if torch.is_tensor(item):
                return item
    return None


class UltralyticsBackboneAdapter(nn.Module):
    """Adapter that exposes Ultralytics YOLO backbone features for CenterNet3D.

    It keeps the existing detection neck/head unchanged by returning a list of
    feature maps and a ``channels`` attribute compatible with DLAUp.
    """

    def __init__(self, model_path, feature_strides=None, feature_indices=None, freeze=False):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError('ultralytics is required for YOLO backbone support.') from e

        if not model_path:
            raise ValueError('model_path is required for Ultralytics backbone.')

        yolo_model = YOLO(model_path)
        self.model = yolo_model.model
        self.feature_strides = feature_strides or [4, 8, 16, 32]
        self.feature_indices = list(feature_indices) if feature_indices is not None else None

        self._feature_cache = {}
        self._hooks = []

        if self.feature_indices is None:
            self.feature_indices, self.channels = self._infer_feature_layers()
        else:
            self.channels = self._infer_channels_from_indices(self.feature_indices)

        self._register_feature_hooks(self.feature_indices)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def _infer_feature_layers(self):
        device = next(self.model.parameters()).device
        dummy = torch.zeros(1, 3, 384, 1280, device=device)

        layer_outputs = {}
        hooks = []
        for idx, layer in enumerate(self.model.model):
            hooks.append(layer.register_forward_hook(lambda _, __, out, i=idx: layer_outputs.__setitem__(i, _extract_tensor(out))))

        training = self.model.training
        self.model.eval()
        with torch.no_grad():
            _ = self.model(dummy)
        if training:
            self.model.train()

        for h in hooks:
            h.remove()

        candidates = {}
        for idx, feat in layer_outputs.items():
            if feat is None or feat.dim() != 4:
                continue
            stride_h = int(round(dummy.shape[-2] / feat.shape[-2]))
            stride_w = int(round(dummy.shape[-1] / feat.shape[-1]))
            if stride_h != stride_w:
                continue
            stride = stride_h
            if stride in self.feature_strides:
                candidates.setdefault(stride, []).append((idx, feat.shape[1]))

        indices, channels = [], []
        for stride in self.feature_strides:
            if stride not in candidates or len(candidates[stride]) == 0:
                raise RuntimeError(f'Cannot infer feature with stride {stride} from Ultralytics model.')
            idx, ch = candidates[stride][-1]
            indices.append(idx)
            channels.append(ch)

        return indices, channels

    def _infer_channels_from_indices(self, feature_indices):
        device = next(self.model.parameters()).device
        dummy = torch.zeros(1, 3, 384, 1280, device=device)
        layer_outputs = {}
        hooks = []
        for idx in feature_indices:
            hooks.append(self.model.model[idx].register_forward_hook(
                lambda _, __, out, i=idx: layer_outputs.__setitem__(i, _extract_tensor(out))
            ))

        training = self.model.training
        self.model.eval()
        with torch.no_grad():
            _ = self.model(dummy)
        if training:
            self.model.train()

        for h in hooks:
            h.remove()

        channels = []
        for idx in feature_indices:
            feat = layer_outputs.get(idx, None)
            if feat is None:
                raise RuntimeError(f'Failed to collect feature map at layer index {idx}.')
            channels.append(feat.shape[1])
        return channels

    def _register_feature_hooks(self, feature_indices):
        for h in self._hooks:
            h.remove()
        self._hooks = []

        for idx in feature_indices:
            layer = self.model.model[idx]
            self._hooks.append(layer.register_forward_hook(
                lambda _, __, out, i=idx: self._feature_cache.__setitem__(i, _extract_tensor(out))
            ))

    def forward(self, x):
        self._feature_cache = {}
        _ = self.model(x)
        features = []
        for idx in self.feature_indices:
            feat = self._feature_cache.get(idx, None)
            if feat is None:
                raise RuntimeError(f'Missing feature map from layer index {idx}.')
            features.append(feat)
        return features
