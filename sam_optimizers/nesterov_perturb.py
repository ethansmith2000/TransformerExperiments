from typing import Generator

import torch


class NesterovPerturb(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=1e-4,
        perturb_lr=None,
        beta1=0.90,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        nesterov_as_perturb=False,
    ):
        perturb_lr = perturb_lr or lr
        defaults = dict(
            lr=lr,
            perturb_lr=perturb_lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            nesterov_as_perturb=nesterov_as_perturb
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

                if state["step"] > 1:
                    # remove last weight decay perturbation
                    perturb = grad.lerp_(state["exp_avg"], group["beta1"]) if group["nesterov_as_perturb"] else state["exp_avg"]
                    denom = state["exp_avg_sq"].sqrt().add_(group["eps"])
                    param.data.addcdiv_(perturb, denom, value=-group["lr"]/scale)
                ############################################################

                # do Adam update
                og_shape = grad.shape

                # momentum update   
                state["exp_avg"].lerp_(grad, 1 - group["beta1"])

                # exp avg sq update
                state["exp_avg_sq"].lerp_(grad.pow(2), 1 - group["beta2"])

                update = grad.lerp_(state["exp_avg"], group["beta1"]) if not group["nesterov_as_perturb"] else state["exp_avg"]

                # update
                denom = state["exp_avg_sq"].sqrt().add_(group["eps"])
                param.data.addcdiv_(update, denom, value=-group["lr"])

                # weight decay
                param.data.mul_(1 - group["lr"] * group["weight_decay"])

                ############################################################

                perturb = grad.lerp_(state["exp_avg"], group["beta1"]) if group["nesterov_as_perturb"] else state["exp_avg"]

                # Do other momentum perturbation
                param.data.addcdiv_(perturb, denom, value=group["lr"]/scale)


