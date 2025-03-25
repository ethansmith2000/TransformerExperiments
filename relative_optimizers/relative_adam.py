import os
import torch
import torch.distributed as dist


class RelativeAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=0, beta1=0.9, beta2=0.999, eps=1e-8, lr_weight=0.5, param_lr=0.005, param_eps=1e-4, lr_cap=0.01):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2, eps=eps, lr_weight=lr_weight, param_lr=param_lr, param_eps=param_eps, lr_cap=lr_cap)
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(g)
                    state['exp_avg_sq'] = torch.zeros_like(g)

                state['step'] += 1

                # update momentum and variance
                state['exp_avg'].lerp_(g, 1 - group['beta1'])
                state['exp_avg_sq'].lerp_(g.square(), 1 - group['beta2'])

                # the update
                g = state['exp_avg'] / (group['eps'] + state['exp_avg_sq'].sqrt())

                # bias correction
                bias_correction1 = 1 - group['beta1'] ** state['step']
                bias_correction2 = 1 - group['beta2'] ** state['step']
                scale = bias_correction1 / bias_correction2**0.5

                # apply weight decay and update
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # regular update
                p.data.add_(g, alpha=-(group['lr'] * group['lr_weight']) / scale)

                # parameter-level learning rate
                p.data.add_(torch.clamp(g * (p.abs() + group['param_eps']), max=group['lr_cap']), alpha=-(group['param_lr'] * (1-group['lr_weight'])) / scale)
