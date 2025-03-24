from typing import Generator
import torch

# https://github.com/KellerJordan/Muon/blob/master/muon.py

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class RelativeMuon(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=0.02,
        beta1=0.95,
        eps=1e-8,
        weight_decay=0.01,
        ns_steps=6,
        exp_avg_momentum=True,
        nesterov=False,
        param_lr=0.005,
        param_eps=1e-4,
        lr_weight=0.5,
        lr_cap=0.01,
    ):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            eps=eps,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            exp_avg_momentum=exp_avg_momentum,
            nesterov=nesterov,
            param_lr=param_lr,
            param_eps=param_eps,
            lr_weight=lr_weight,
            lr_cap=lr_cap,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_num, group in enumerate(self.param_groups):
            for i, param in enumerate(group["params"]):
                grad = param.grad
                if grad is None:
                    continue

                # do Muon update
                og_shape = grad.shape
                if grad.ndim != 2:
                    grad = grad.view(grad.size(0), -1)
                    
                state = self.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["step"] = 0

                state["step"] += 1

                # momentum update   
                if group['exp_avg_momentum']:
                    state["exp_avg"].lerp_(grad, 1 - group["beta1"])
                else:
                    state["exp_avg"].mul_(group["beta1"]).add_(grad)

                update = grad.lerp_(state["exp_avg"], group["beta1"]) if group["nesterov"] else state["exp_avg"]

                # orthogonalization
                g = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])

                # rescaling
                g *= max(1, g.size(0)/g.size(1))**0.5
                g = g.view(og_shape).type_as(param.data)

                # weight decay
                param.data.mul_(1 - group["lr"] * group["weight_decay"])

                # regular update
                param.data.add_(g, alpha=-(group['lr'] * group['lr_weight']))

                # parameter-level learning rate
                param.data.add_(g * torch.clamp(param.abs() + group['param_eps'], max=group['lr_cap']), alpha=-(group['param_lr'] * (1-group['lr_weight'])))



