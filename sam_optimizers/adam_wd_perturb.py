from typing import Generator

import torch


class AdamWeightDecaySAM(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=1e-4,
        beta1=0.90,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,

    ):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
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

                state = self.state[param]

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["step"] = 0

                # do Adam update
                state["step"] += 1

                bias_correction1 = 1 - group["beta1"]**state["step"]
                bias_correction2 = 1 - group["beta2"]**state["step"]
                scale = bias_correction1 / bias_correction2**0.5

                # remove last weight decay perturbation, 
                if state["step"] > 1:
                    param.data.div_(1 - group["lr"] * group["weight_decay"])
                ############################################################

                # do Adam update
                og_shape = grad.shape
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["step"] = 0


                # momentum update   
                state["exp_avg"].lerp_(grad, 1 - group["momentum"])

                # exp avg sq update
                state["exp_avg_sq"].lerp_(grad.pow(2), 1 - group["beta2"])

                # update and weight decay
                denom = state["exp_avg_sq"].sqrt().add_(group["eps"])
                param.data.addcdiv_(state["exp_avg"], denom, value=-group["lr"]/scale)
                param.data.mul_(1 - group["lr"] * group["weight_decay"])

                ############################################################

                # Do weight decay perturbation
                param.data.mul_(1 - group["lr"] * group["weight_decay"])


