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

class CautionAdamMuon(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=0.02,
        beta1=0.95,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        ns_steps=6,
        nesterov=False,
        update_type="adam",
        caution_mode="caution",
    ):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            nesterov=nesterov,
            update_type=update_type,
            caution_mode=caution_mode,
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

                og_shape = grad.shape
                if grad.ndim != 2:
                    grad = grad.view(grad.size(0), -1)

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["step"] = 0

                # do Adam update
                state["step"] += 1

                bias_correction1 = 1 - group["beta1"]**state["step"]
                bias_correction2 = 1 - group["beta2"]**state["step"]
                scale = bias_correction1 / bias_correction2**0.5

                # first and second moment update
                state["exp_avg"].lerp_(grad, 1 - group["beta1"])
                state["exp_avg_sq"].lerp_(grad.pow(2), 1 - group["beta2"])

                muon_update = grad.lerp_(state["exp_avg"], group["beta1"]) if group["nesterov"] else state["exp_avg"]

                # orthogonalization
                muon_update = zeropower_via_newtonschulz5(muon_update, steps=group["ns_steps"])

                # rescaling
                muon_update *= max(1, muon_update.size(0)/muon_update.size(1))**0.5
                # muon_update = muon_update.view(og_shape).type_as(param.data)

                # adam update
                denom = state["exp_avg_sq"].sqrt().add_(group["eps"])
                adam_update = state["exp_avg"].div(denom)

                # weight decay
                param.data.mul_(1 - group["lr"] * group["weight_decay"])

                # caution mask
                if group["caution_mode"] == "caution":
                    mask = (adam_update * muon_update > 0).to(muon_update.dtype).squeeze()
                    update = muon_update if group["update_type"] == "muon" else adam_update
                    param.data.add_(update.squeeze() * mask/(mask.mean()+group["eps"]), alpha=-group["lr"])
                elif group["caution_mode"] == "scaling":                    
                    # Calculate cosine similarity using torch's function (faster)
                    cosine_sim = torch.nn.functional.cosine_similarity(adam_update.flatten().unsqueeze(0), muon_update.flatten().unsqueeze(0)).item()
                    
                    # Scale factor based on cosine similarity (higher similarity = higher scale)
                    # When cosine_sim is 1, scale is 1; when cosine_sim is -1, scale is close to 0
                    scale_factor = (cosine_sim + 1)  # Map from [-1,1] to [0,2]
                    
                    # Apply the scaling to the update
                    update = muon_update if group["update_type"] == "muon" else adam_update
                    param.data.add_(update * scale_factor, alpha=-group["lr"])



