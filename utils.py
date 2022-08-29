import torch

class FeatureNormL2(torch.nn.Module):
    def __call__(self, x):
        return torch.nn.functional.normalize(x)