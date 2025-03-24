import os
import torch
import torch.distributed as dist


class RelativeAdam2(torch.optim.Optimizer):
    def __init__(self, params, param_lr=1e-4, weight_decay=0, beta1=0.9, beta2=0.999, eps=1e-8, param_eps=1e-4, lr_cap=0.01):
        defaults = dict(param_lr=param_lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2, eps=eps, param_eps=param_eps, lr_cap=lr_cap)

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
                p.data.mul_(1 - group['param_lr'] * group['weight_decay'])

                # parameter-level learning rate
                p.data.add_(g * torch.clamp(p.abs() + group['param_eps'], max=group['lr_cap']), alpha=-group['param_lr'] / scale)




