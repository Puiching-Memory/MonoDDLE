import numpy as np
import torch


def laplacian_aleatoric_uncertainty_loss(input, target, log_variance, reduction='mean'):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss = 1.4142 * torch.exp(-log_variance) * torch.abs(input - target) + log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()


def gaussian_aleatoric_uncertainty_loss(input, target, log_variance, reduction='mean'):
    '''
    References:
        What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, Neuips'17
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss = 0.5 * torch.exp(-log_variance) * torch.abs(input - target)**2 + 0.5 * log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()


def kendall_uncertainty_distill_loss(pred, teacher, log_variance, weight=None):
    """Kendall's uncertainty loss for noise-robust depth distillation.

    When the network detects that DA3's pseudo-label contradicts the image
    features, it predicts a large variance to down-weight that pixel's penalty,
    achieving robust distillation against noisy teacher labels.

    L = ||d_pred - d_DA3||^2 / (2 * sigma^2) + 0.5 * log(sigma^2)

    where log_variance = log(sigma^2), so sigma^2 = exp(log_variance).

    Args:
        pred: Predicted depth, shape (B, 1, H, W).
        teacher: DA3 teacher depth, shape (B, 1, H, W).
        log_variance: Predicted log-variance, shape (B, 1, H, W).
        weight: Per-pixel weight mask, shape (B, 1, H, W) or None.

    Returns:
        Scalar loss value.

    References:
        Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep
        Learning for Computer Vision?", NeurIPS 2017.
    """
    residual_sq = (pred - teacher) ** 2
    # Kendall's heteroscedastic uncertainty formulation
    loss = 0.5 * torch.exp(-log_variance) * residual_sq + 0.5 * log_variance

    if weight is not None:
        loss = loss * weight
        return loss.sum() / weight.sum().clamp(min=1.0)
    return loss.mean()



if __name__ == '__main__':
    pass
