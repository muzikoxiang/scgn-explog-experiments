"""Tensor-only equivalents of upstream normalization and constant padding."""
import torch
import torch.nn.functional as F


class TF:
    @staticmethod
    def pad(x, padding):
        left, top, right, bottom = padding
        return F.pad(x, (left, right, top, bottom))

    @staticmethod
    def normalize(x, mean, std):
        shape = (1, -1, 1, 1)
        return (x - x.new_tensor(mean).view(shape)) / x.new_tensor(std).view(shape)


def denormalize(x, mean, std):
    return x * x.new_tensor(std).view(1, -1, 1, 1) + x.new_tensor(mean).view(1, -1, 1, 1)


def mean_std(scale, target_mode):
    if scale != 1 or target_mode != 'light_graydn':
        raise ValueError('This package prepares grayscale denoising only')
    return (0.4539,), (0.2326,)


def register_notrace_function(function):
    return function
