from typing import Generator

import torch


class AdamTwoMomentumSAM(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=1e-4,
        perturb_lr_ratio=None,
        beta1=0.90,
        beta1_perturb=0.80,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        nesterov=False,
        perturbation_start_step=50,
    ):
        perturb_lr_ratio = perturb_lr_ratio or 1.0
        defaults = dict(
            lr=lr,
            perturb_lr_ratio=perturb_lr_ratio,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            beta1_perturb=beta1_perturb,
            nesterov=nesterov,
            perturbation_start_step=perturbation_start_step,
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
                    state["exp_avg_perturb"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["step"] = 0

                # do Adam update
                state["step"] += 1

                bias_correction1 = 1 - group["beta1"]**state["step"]
                bias_correction2 = 1 - group["beta2"]**state["step"]
                scale = bias_correction1 / bias_correction2**0.5

                if state["step"] > 1 and state["step"] > group["perturbation_start_step"]:
                    # remove last weight decay perturbation
                    perturb_lr = group["lr"] * group["perturb_lr_ratio"]
                    denom = state["exp_avg_sq"].sqrt().add_(group["eps"])
                    param.data.addcdiv_(state["exp_avg_perturb"], denom, value=-perturb_lr/scale)
                ############################################################

                # momentum update   
                state["exp_avg"].lerp_(grad, 1 - group["beta1"])
                state["exp_avg_perturb"].lerp_(grad, 1 - group["beta1_perturb"])

                # exp avg sq update
                state["exp_avg_sq"].lerp_(grad.pow(2), 1 - group["beta2"])

                update = grad.lerp_(state["exp_avg"], group["beta1"]) if group["nesterov"] else state["exp_avg"]

                # update
                denom = state["exp_avg_sq"].sqrt().add_(group["eps"])
                param.data.addcdiv_(update, denom, value=-group["lr"])

                # weight decay
                param.data.mul_(1 - group["lr"] * group["weight_decay"])

                ############################################################

                # Do other momentum perturbation
                if state["step"] > group["perturbation_start_step"]:
                    perturb_lr = group["lr"] * group["perturb_lr_ratio"]
                    param.data.addcdiv_(state["exp_avg_perturb"], denom, value=perturb_lr/scale)


