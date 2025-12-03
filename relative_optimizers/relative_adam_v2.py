import torch


class RelativeAdamV2(torch.optim.Optimizer):
    """
    RelativeAdam with percentage-based updates.
    
    Instead of absolute updates, applies updates as a percentage of current parameter value.
    This naturally handles parameters at different scales without explosion.
    
    Args:
        param_lr: Percentage of parameter value to update per step (e.g., 0.01 = 1% update)
        This replaces the problematic g * p multiplication with p * percentage,
        where percentage is derived from the normalized gradient direction.
    """
    def __init__(
        self,
        params,
        lr=1e-4,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        lr_weight=0.5,
        param_lr=0.01,  # Now interpreted as max percentage change
        param_eps=1e-4,  # Minimum absolute value to apply relative updates
    ):
        defaults = dict(
            lr=lr,
            orig_lr=lr,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            lr_weight=lr_weight,
            param_lr=param_lr,
            param_eps=param_eps,
        )
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

                # Compute normalized gradient (Adam update direction)
                g_normalized = state['exp_avg'] / (group['eps'] + state['exp_avg_sq'].sqrt())

                # bias correction
                bias_correction1 = 1 - group['beta1'] ** state['step']
                bias_correction2 = 1 - group['beta2'] ** state['step']
                scale = bias_correction1 / bias_correction2**0.5

                # apply weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Component 1: Regular Adam update (absolute)
                p.data.add_(g_normalized, alpha=-(group['lr'] * group['lr_weight']) / scale)

                # Component 2: Relative update (percentage-based)
                # Update each parameter by a percentage of its current value
                # Direction comes from normalized gradient, magnitude from param_lr
                
                # to handle lr scheduling
                ratio = group['lr'] / group['orig_lr']
                param_lr = group['param_lr'] * ratio
                
                # Percentage-based update: direction × percentage × current_value
                # For large parameters, update is large. For small parameters, update is small.
                # But crucially: doesn't depend on gradient magnitude!
                percentage_update = g_normalized * p.abs() * (param_lr * (1 - group['lr_weight']))
                
                # Only apply relative updates to parameters above threshold
                # (avoid numerical issues with very small parameters)
                mask = p.abs() > group['param_eps']
                p.data.add_(
                    torch.where(mask, percentage_update, torch.zeros_like(percentage_update)),
                    alpha=-1.0 / scale
                )

