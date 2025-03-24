from typing import Generator
from .utils import zeropower_via_newtonschulz5
import torch

# https://github.com/KellerJordan/Muon/blob/master/muon.py



class Muon(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=0.02,
        beta1=0.95,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        ns_steps=6,
        exp_avg_momentum=True,
        nesterov=False,
    ):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            exp_avg_momentum=exp_avg_momentum,
            nesterov=nesterov
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
                state = self.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["step"] = 0

                state["step"] += 1

                if grad.ndim > 2:
                    grad = grad.view(grad.size(0), -1)

                # momentum update   
                if group['exp_avg_momentum']:
                    state["exp_avg"].lerp_(grad, 1 - group["momentum"])
                else:
                    state["exp_avg"].mul_(group["momentum"]).add_(grad)

                update = grad.lerp_(state["exp_avg"], group["beta1"]) if group["nesterov"] else state["exp_avg"]

                # orthogonalization
                g = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])

                # rescaling
                g *= max(1, g.size(0)/g.size(1))**0.5
                g = g.view(og_shape).type_as(param.data)

                # update and weight decay
                param.data.add_(g, alpha=-group["lr"])
                param.data.mul_(1 - group["lr"] * group["weight_decay"])



