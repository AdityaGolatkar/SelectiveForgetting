import torch
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import types

import torch.nn as nn


def get_variational_vars(model):
    """Returns all variables involved in optimizing the hessian estimation."""
    result = []
    if hasattr(model, 'logvar0'):
        result.append(model.logvar0)
        result.append(model.loglambda2)
    for l in model.children():
        result += get_variational_vars(l)
    return result


def get_compression_loss(model):
    """Get the model loss function for hessian estimation."""
    # Compute quantity of information in the in weights
    # Assuming a normal prior p(w) ~ N(0, lambda**2 * I)
    # (where lambda is selected independently for each layer)
    # we have
    #    I(w; D) ~= KL(q(w|D) | p(w))
    # where the optimal posterior q(w|D) is q(w|D) ~ N(w, F**-1)
    # and F is the Fisher information matrix

    modules = [x for x in model.modules() if hasattr(x, 'logvar0')]
    k = sum([x.weight.numel() for x in modules])

    w_norm2 = sum([x.weight.pow(2).sum() / x.loglambda2.exp() for x in modules])
    logvar = sum([x.logvar.sum() for x in modules])
    trace = sum([x.logvar.exp().sum() / x.loglambda2.exp() for x in modules])
    lambda2_cost = sum([x.loglambda2 * x.weight.numel() for x in modules])

    # Standard formula for KL divergence of two normal distributions
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
    Lz = kl_divergence = w_norm2 + trace + lambda2_cost - logvar - k
    return Lz
    # return -logvar


def variational_forward(module, input):
    """Modified forward pass that adds noise to the output."""
    # Recall that module.logvar0 is created by make_variational()
    # (specifically, by add_logvar())

    module.logvar = module.logvar0.expand_as(module.weight).clamp(max=-3.0)

    var = module.logvar.exp()

    if isinstance(module, torch.nn.modules.conv.Conv2d):
        output = F.conv2d(input, module.weight, module.bias, module.stride,
                          module.padding, module.dilation, module.groups)
        # From Variational Dropout and the Local reparametrization trick
        # (Kingma et al., 2015)
        output_var = F.conv2d(input ** 2 + 1e-5, var, None, module.stride,
                              module.padding, module.dilation, module.groups)
    elif isinstance(module, torch.nn.modules.linear.Linear):
        output = F.linear(input, module.weight, module.bias)
        output_var = F.linear(input ** 2 + 1e-5, var, None)
    else:
        raise NotImplementedError(f"Module {type(module)} not implemented.")

    eps = output.data.clone().normal_()
    # print(f"max: {var.max().item():.4f} min: {var.min().item():.4f} nan: {torch.isnan(var).sum()}")
    # if output_var.min().item() < 0:
    #     raise ValueError('Variance less than 0.')
    # Local reparemetrization trick
    if module.disable_noise:
        return output
    else:
        return output + torch.sqrt(output_var) * eps


def _reset_logvar(module, variance_scaling=0.005):
    if hasattr(module, 'logvar0'):
        w = module.weight.data
        # Initial ballpark estimate for optimal variance is the variance
        # of the weights in the kernel
        var = w.view(w.size(0), -1).var(dim=1).view(-1, *([1] * (w.ndimension() - 1)))  # .expand_as(w)
        # Further scale down the variance by some factor
        module.logvar0.data[:] = (var * variance_scaling + 1e-8).log()
        # Initial guess for lambda is the l2 norm of the weights
        module.loglambda2.data = (w.pow(2).mean() + 1e-8).log()


def _add_logvar(module):
    """Adds a parameter (logvar0) to store the noise variance for the weights.

    Also adds a scalar parameter loglambda2 to store the scaling coefficient
    for the layer.

    The variance is assumed to be the same for all weights in the same filter.
    The common value is stored in logvar0, which is expanded to the same
    dimension as the weight matrix in logvar.
    """
    if not hasattr(module, 'weight'):
        return
    if module.weight.data.ndimension() < 2:
        return
    if not hasattr(module, 'logvar0'):
        w = module.weight.data
        # w is of shape NUM_OUT x NUM_IN x K_h X K_w
        var = w.view(w.size(0), -1).var(dim=1).view(-1, *([1] * (w.ndimension() - 1)))
        # var is of shape NUM_OUT x 1 x 1 x 1
        # (so that it can be expanded to the same size as w by torch.expand_as())
        # The content does not matter since we will reset it later anyway
        module.logvar0 = Parameter(var.log())
        # log(lambda**2) is a scalar shared by all weights in the layer
        module.loglambda2 = Parameter(w.pow(2).mean().log())
        module.logvar = module.logvar0.expand_as(module.weight)
        _reset_logvar(module)


def make_variational(model):
    """Replaces the forward pass of the model layers to add noise."""
    model.apply(_add_logvar)
    for m in model.modules():
        if hasattr(m, 'logvar0'):
            m.forward = types.MethodType(variational_forward, m)
            m.disable_noise = False
