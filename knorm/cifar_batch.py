
from train_cifar10 import train_model, default_args
from copy import deepcopy

runs = [
    # dict(acts=["gelu", "gelu", "gelu", "gelu", "gelu", "gelu"]),
    # dict(acts=["gelu", "gelu", "gelu", "gelu", "gelu", "gelu"], act_powers=[2, 2, 2, 2, 2, 2]),
    # dict(acts=["leaky", "leaky", "leaky", "leaky", "leaky", "leaky"], act_powers=[3, 3, 3, 3, 3, 3]),
    # dict(acts=["relu", "relu", "relu", "relu", "relu", "relu"]),
    # dict(acts=["relu", "relu", "relu", "relu", "relu", "relu"], act_powers=[2, 2, 2, 2, 2, 2]),
    # dict(acts=["relu", "relu", "relu", "relu", "relu", "relu"], act_powers=[3, 3, 3, 3, 3, 3]),
    # dict(acts=["relu_sin", "relu_sin", "relu_sin", "relu_sin", "relu_sin", "relu_sin"]),

    dict(acts=["gelu", "gelu", "gelu", "gelu", "gelu", "gelu"], val_act="gelu"),
    dict(acts=["gelu", "gelu", "gelu", "gelu", "gelu", "gelu"], act_powers=[2, 2, 2, 2, 2, 2], val_act="gelu"),
    dict(acts=["leaky", "leaky", "leaky", "leaky", "leaky", "leaky"], act_powers=[3, 3, 3, 3, 3, 3], val_act="gelu"),
    dict(acts=["relu", "relu", "relu", "relu", "relu", "relu"], val_act="gelu"),
    dict(acts=["relu", "relu", "relu", "relu", "relu", "relu"], act_powers=[2, 2, 2, 2, 2, 2], val_act="gelu"),
    dict(acts=["relu", "relu", "relu", "relu", "relu", "relu"], act_powers=[3, 3, 3, 3, 3, 3], val_act="gelu"),

    dict(acts=["gelu", "gelu", "gelu", "gelu", "gelu", "gelu"], val_act="gelu", attn_power=2),
    dict(acts=["gelu", "gelu", "gelu", "gelu", "gelu", "gelu"], act_powers=[2, 2, 2, 2, 2, 2], val_act="gelu", attn_power=2),
    dict(acts=["leaky", "leaky", "leaky", "leaky", "leaky", "leaky"], act_powers=[3, 3, 3, 3, 3, 3], val_act="gelu", attn_power=2),
    dict(acts=["relu", "relu", "relu", "relu", "relu", "relu"], val_act="gelu", attn_power=2),
    dict(acts=["relu", "relu", "relu", "relu", "relu", "relu"], act_powers=[2, 2, 2, 2, 2, 2], val_act="gelu", attn_power=2),
    dict(acts=["relu", "relu", "relu", "relu", "relu", "relu"], act_powers=[3, 3, 3, 3, 3, 3], val_act="gelu", attn_power=2),

    dict(acts=["gelu", "gelu", "gelu", "gelu", "gelu", "gelu"], val_act="leaky", attn_power=3),
    dict(acts=["gelu", "gelu", "gelu", "gelu", "gelu", "gelu"], act_powers=[2, 2, 2, 2, 2, 2], val_act="leaky", attn_power=3),
    dict(acts=["leaky", "leaky", "leaky", "leaky", "leaky", "leaky"], act_powers=[3, 3, 3, 3, 3, 3], val_act="leaky", attn_power=3),
    dict(acts=["relu", "relu", "relu", "relu", "relu", "relu"], val_act="leaky", attn_power=3),
    dict(acts=["relu", "relu", "relu", "relu", "relu", "relu"], act_powers=[2, 2, 2, 2, 2, 2], val_act="leaky", attn_power=3),
    dict(acts=["relu", "relu", "relu", "relu", "relu", "relu"], act_powers=[3, 3, 3, 3, 3, 3], val_act="leaky", attn_power=3),
]

for run in runs:
    args = deepcopy(default_args)
    args.update(run)
    train_model(args)
    
