import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torch.optim import Adam

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from copy import deepcopy
from types import SimpleNamespace
import datetime
import wandb
os.makedirs('logs', exist_ok=True)


# cifar traininig data
def cifar_dataloader(batch_size, train=True, num_workers=8):
    transform = []
    if train:
        transform.extend([
            transforms.RandAugment(2, 14),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    transform.extend([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform = transforms.Compose(transform)
    # augs
    train_loader = DataLoader(datasets.CIFAR10('data', 
                                train=train, 
                                download=True, 
                                transform=transform),
                              batch_size=batch_size, 
                              shuffle=train,
                              num_workers=num_workers,
                              )
    return train_loader


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., out_dim=None):
        super().__init__()
        out_dim = out_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., out_dim=None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        out_dim = out_dim or dim

        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., 
            scale_residual_attn=False,
            scale_residual_stream_attn=False,
            scale_residual_ff=False,
            scale_residual_stream_ff=False,
            residual_stream_base_val=1,
            residual_stream_act_fn=Identity,
            residual_base_val=1,
            residual_act_fn=Identity,
            ):
        super().__init__()
        self.attentions = nn.ModuleList([])
        self.attn_norms = nn.ModuleList([])
        self.ffs = nn.ModuleList([])
        self.ff_norms = nn.ModuleList([])
        self.residual_act_fn = residual_act_fn()
        self.residual_stream_act_fn = residual_stream_act_fn()
        self.residual_base_val = residual_base_val
        self.residual_stream_base_val = residual_stream_base_val

        out_dim_mult_attn = 1 + sum([scale_residual_attn, scale_residual_stream_attn])
        out_dim_mult_ff = 1 + sum([scale_residual_ff, scale_residual_stream_ff])
        for _ in range(depth):
            self.attentions.append(Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, out_dim = dim * out_dim_mult_attn))
            self.attn_norms.append(nn.LayerNorm(dim))
            self.ffs.append(FeedForward(dim, mlp_dim, dropout = dropout, out_dim = dim * out_dim_mult_ff))
            self.ff_norms.append(nn.LayerNorm(dim))

        if scale_residual_attn and scale_residual_stream_attn:
            self.update_residual_attn = self.update_scale_residual_and_residual_stream
        elif scale_residual_attn:
            self.update_residual_attn = self.update_scale_residual
        elif scale_residual_stream_attn:
            self.update_residual_attn = self.update_residual_stream
        else:
            self.update_residual_attn = lambda x, y: x + y

        if scale_residual_ff and scale_residual_stream_ff:
            self.update_residual_ff = self.update_scale_residual_and_residual_stream
        elif scale_residual_ff:
            self.update_residual_ff = self.update_scale_residual
        elif scale_residual_stream_ff:
            self.update_residual_ff = self.update_residual_stream
        else:
            self.update_residual_ff = lambda x, y: x + y

    def update_scale_residual_and_residual_stream(self, out, x):
        residual, gate_stream, gate_residual = out.chunk(3, dim = -1)
        return x * self.residual_stream_act_fn(gate_stream + self.residual_stream_base_val) + residual * self.residual_act_fn(gate_residual + self.residual_base_val)

    def update_scale_residual(self, out, x):
        residual, gate_residual = out.chunk(2, dim = -1)
        return x + residual * self.residual_act_fn(gate_residual + self.residual_base_val)

    def update_residual_stream(self, out, x):
        residual, gate_stream = out.chunk(2, dim = -1)
        return x * self.residual_stream_act_fn(gate_stream + self.residual_stream_base_val) + residual

    def forward(self, x):
        for attn, attn_norm, ff, ff_norm in zip(self.attentions, self.attn_norms, self.ffs, self.ff_norms):
            x = self.update_residual_attn(attn(attn_norm(x)), x)
            x = self.update_residual_ff(ff(ff_norm(x)), x)
        return x


class ViT(nn.Module):
    def __init__(self, image_size=32,
                        patch_size=2,
                        num_classes=10,
                        dim=512,
                        depth=6,
                        heads=8,
                        mlp_dim=512,
                        channels = 3,
                        dim_head = 64, 
                        dropout = 0.1, 
                        emb_dropout = 0.1,
                        scale_residual_attn=False,
                        scale_residual_stream_attn=False,
                        scale_residual_ff=False,
                        scale_residual_stream_ff=False,
                        residual_stream_base_val=1,
                        residual_stream_act_fn=Identity,
                        residual_base_val=1,
                        residual_act_fn=Identity,
                        ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, 
                                        depth, 
                                        heads, 
                                        dim_head, 
                                        mlp_dim, 
                                        dropout,
                                        )

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


# training

if __name__ == '__main__':

    base_args = SimpleNamespace(
        batch_size=512,
        epochs=200,
        lr=2e-4,
        rank=8,
        weight_decay=1e-3,
        beta_1=0.9,
        beta_2=0.998,
        validate_every=1,
        save_every=100000,
        use_fp16=True,
        compile=True,
        max_grad_norm=None,
        lr_warmp_up_steps=50,

        scale_residual_attn=False,
        scale_residual_stream_attn=False,
        scale_residual_ff=False,
        scale_residual_stream_ff=False,
        residual_stream_base_val=1,
        residual_stream_act_fn=Identity,
        residual_base_val=1,
        residual_act_fn=Identity,
        run_name = "baseline",
    )

    runs = [
        # # {
        # #     "run_name": "baseline",
        # # },
        # {
        #     'scale_residual_attn': True,
        #     "run_name": "resid_attn",
        # },
        # {
        #     'scale_residual_stream_attn': True,
        #     "run_name": "stream_attn",
        # },
        # {
        #     'scale_residual_ff': True,
        #     "run_name": "resid_ff",
        # },
        # {
        #     'scale_residual_stream_ff': True,
        #     "run_name": "stream_ff",
        # },
        # {
        #     'scale_residual_attn': True,
        #     'scale_residual_stream_attn': True,
        #     "run_name": "resid+stream_attn",
        # },
        # {
        #     'scale_residual_ff': True,
        #     'scale_residual_stream_ff': True,
        #     "run_name": "resid+stream_ff",
        # },
        # {
        #     'scale_residual_attn': True,
        #     'scale_residual_stream_attn': True,
        #     'scale_residual_ff': True,
        #     'scale_residual_stream_ff': True,
        #     "run_name": "all",
        # },



        # {
        #     'scale_residual_attn': True,
        #     'residual_base_val': 0,
        #     "run_name": "resid_attn_gate0",
        # },
        # {
        #     'scale_residual_ff': True,
        #     'residual_base_val': 0,
        #     "run_name": "resid_ff_gate0",
        # },
        # {
        #     'scale_residual_attn': True,
        #     'scale_residual_stream_attn': True,
        #     'residual_base_val': 0,
        #     "run_name": "resid+stream_attn_gate0",
        # },
        # {
        #     'scale_residual_ff': True,
        #     'scale_residual_stream_ff': True,
        #     'residual_base_val': 0,
        #     "run_name": "resid+stream_ff_gate0",
        # },
        # {
        #     'scale_residual_attn': True,
        #     'scale_residual_stream_attn': True,
        #     'scale_residual_ff': True,
        #     'scale_residual_stream_ff': True,
        #     'residual_base_val': 0,
        #     "run_name": "all_gate0",
        # },



        # {
        #     'scale_residual_stream_attn': True,
        #     'residual_stream_base_val': 0,
        #     "run_name": "stream_attn_gate0",
        # },
        # {
        #     'scale_residual_stream_ff': True,
        #     'residual_stream_base_val': 0,
        #     "run_name": "stream_ff_gate0",
        # },
        # {
        #     'scale_residual_attn': True,
        #     'scale_residual_stream_attn': True,
        #     'residual_stream_base_val': 0,
        #     "run_name": "resid+stream_attn_gate0",
        # },
        # {
        #     'scale_residual_ff': True,
        #     'scale_residual_stream_ff': True,
        #     'residual_stream_base_val': 0,
        #     "run_name": "resid+stream_ff_gate0",
        # },
        # {
        #     'scale_residual_attn': True,
        #     'scale_residual_stream_attn': True,
        #     'scale_residual_ff': True,
        #     'scale_residual_stream_ff': True,
        #     'residual_stream_base_val': 0,
        #     "run_name": "all_gate0",
        # },



        # {
        #     'scale_residual_attn': True,
        #     'scale_residual_stream_attn': True,
        #     'residual_base_val': 0,
        #     'residual_stream_base_val': 0,
        #     "run_name": "resid+stream_attn_gate0both",
        # },
        # {
        #     'scale_residual_ff': True,
        #     'scale_residual_stream_ff': True,
        #     'residual_base_val': 0,
        #     'residual_stream_base_val': 0,
        #     "run_name": "resid+stream_ff_gate0both",
        # },
        # {
        #     'scale_residual_attn': True,
        #     'scale_residual_stream_attn': True,
        #     'scale_residual_ff': True,
        #     'scale_residual_stream_ff': True,
        #     'residual_base_val': 0,
        #     'residual_stream_base_val': 0,
        #     "run_name": "all_gate0both",
        # },




        # {
        #     'scale_residual_attn': True,
        #     'residual_base_val': 0,
        #     'residual_act_fn': torch.nn.ReLU,
        #     "run_name": "resid_attn_gate0_relu",
        # },
        # {
        #     'scale_residual_ff': True,
        #     'residual_base_val': 0,
        #     'residual_act_fn': torch.nn.ReLU,
        #     "run_name": "resid_ff_gate0_relu",
        # },
        # {
        #     'scale_residual_attn': True,
        #     'scale_residual_stream_attn': True,
        #     'residual_base_val': 0,
        #     'residual_act_fn': torch.nn.ReLU,
        #     "run_name": "resid+stream_attn_gate0_relu",
        # },
        # {
        #     'scale_residual_ff': True,
        #     'scale_residual_stream_ff': True,
        #     'residual_base_val': 0,
        #     'residual_act_fn': torch.nn.ReLU,
        #     "run_name": "resid+stream_ff_gate0_relu",
        # },
        # {
        #     'scale_residual_attn': True,
        #     'scale_residual_stream_attn': True,
        #     'scale_residual_ff': True,
        #     'scale_residual_stream_ff': True,
        #     'residual_base_val': 0,
        #     'residual_act_fn': torch.nn.ReLU,
        #     "run_name": "all_gate0_relu",
        # },



        # {
        #     'scale_residual_stream_attn': True,
        #     'residual_stream_base_val': 0,
        #     'residual_stream_act_fn': torch.nn.ReLU,
        #     "run_name": "stream_attn_gate0_relu",
        # },
        # {
        #     'scale_residual_stream_ff': True,
        #     'residual_stream_base_val': 0,
        #     'residual_stream_act_fn': torch.nn.ReLU,
        #     "run_name": "stream_ff_gate0_relu",
        # },
        # {
        #     'scale_residual_attn': True,
        #     'scale_residual_stream_attn': True,
        #     'residual_stream_base_val': 0,
        #     'residual_stream_act_fn': torch.nn.ReLU,
        #     "run_name": "resid+stream_attn_gate0_relu",
        # },
        # {
        #     'scale_residual_ff': True,
        #     'scale_residual_stream_ff': True,
        #     'residual_stream_base_val': 0,
        #     'residual_stream_act_fn': torch.nn.ReLU,
        #     "run_name": "resid+stream_ff_gate0_relu",
        # },
        # {
        #     'scale_residual_attn': True,
        #     'scale_residual_stream_attn': True,
        #     'scale_residual_ff': True,
        #     'scale_residual_stream_ff': True,
        #     'residual_stream_base_val': 0,
        #     'residual_stream_act_fn': torch.nn.ReLU,
        #     "run_name": "all_gate0_relu",
        # },

        # {
        #     'scale_residual_attn': True,
        #     'scale_residual_stream_attn': True,
        #     'residual_base_val': 0,
        #     'residual_stream_base_val': 0,
        #     'residual_act_fn': torch.nn.ReLU,
        #     "run_name": "resid+stream_attn_gate0both_relu",
        # },
        # {
        #     'scale_residual_ff': True,
        #     'scale_residual_stream_ff': True,
        #     'residual_base_val': 0,
        #     'residual_stream_base_val': 0,
        #     'residual_act_fn': torch.nn.ReLU,
        #     "run_name": "resid+stream_ff_gate0both_relu",
        # },
        # {
        #     'scale_residual_attn': True,
        #     'scale_residual_stream_attn': True,
        #     'scale_residual_ff': True,
        #     'scale_residual_stream_ff': True,
        #     'residual_base_val': 0,
        #     'residual_stream_base_val': 0,
        #     'residual_act_fn': torch.nn.ReLU,
        #     "run_name": "all_gate0both_relu",
        # },


        {
            'scale_residual_attn': True,
            'residual_base_val': 0,
            "run_name": "resid_attn_gate0",
        },
        {
            'scale_residual_ff': True,
            'residual_base_val': 0,
            "run_name": "resid_ff_gate0",
        },
        {
            'scale_residual_attn': True,
            'scale_residual_stream_attn': True,
            'residual_base_val': 0,
            "run_name": "resid+stream_attn_gate0",
        },
        {
            'scale_residual_ff': True,
            'scale_residual_stream_ff': True,
            'residual_base_val': 0,
            "run_name": "resid+stream_ff_gate0",
        },
        {
            'scale_residual_attn': True,
            'scale_residual_stream_attn': True,
            'scale_residual_ff': True,
            'scale_residual_stream_ff': True,
            'residual_base_val': 0,
            "run_name": "all_gate0",
        },
    ]


    for run in runs:
        args = deepcopy(base_args)
        args.__dict__.update(run)
        print(args)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # launch wandb
        wandb.init(project='babyhypernetworks', name=args.run_name)
        wandb.config.update(args)

        train_loader = cifar_dataloader(args.batch_size, train=True)
        val_loader = cifar_dataloader(args.batch_size, train=False)
        model = ViT(
            image_size=32,
            patch_size=4,
            num_classes=10,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            channels = 3,
            dim_head = 64, 
            dropout = 0.1, 
            emb_dropout = 0.1,
            scale_residual_attn=args.scale_residual_attn,
            scale_residual_stream_attn=args.scale_residual_stream_attn,
            scale_residual_ff=args.scale_residual_ff,
            scale_residual_stream_ff=args.scale_residual_stream_ff,
            residual_stream_base_val=args.residual_stream_base_val,
            residual_stream_act_fn=args.residual_stream_act_fn,
            residual_base_val=args.residual_base_val,
            residual_act_fn=args.residual_act_fn,
        ).to(device).train()
        print(sum(p.numel() for p in model.parameters()))
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta_1, args.beta_2))
        criterion = nn.CrossEntropyLoss()

        run_stats = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
        }
        scaler = torch.amp.GradScaler(enabled=args.use_fp16)

        if args.compile:
            model = torch.compile(model)

        # calculte number of steps
        num_steps = len(train_loader) * args.epochs

        # warmup then decay
        def create_lr_lambda(num_steps, lr_warmp_up_steps):
            def lr_lambda(w):
                if w < lr_warmp_up_steps:
                    return min(1,w / lr_warmp_up_steps)
                return max(0, (num_steps - w) / (num_steps - lr_warmp_up_steps))
            return lr_lambda
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, create_lr_lambda(num_steps, args.lr_warmp_up_steps))

        for epoch in range(args.epochs):
            model.train()
            train_losses = []
            pbar = tqdm(train_loader)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(enabled=args.use_fp16, device_type='cuda'):
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                train_losses.append(loss.item())
                pbar.set_description(f'loss: {loss.item()}, lr: {lr_scheduler.get_last_lr()[0]}')
            run_stats['train_loss'].append(np.mean(train_losses))
            wandb.log({'train_loss': run_stats['train_loss'][-1]})
            if epoch % args.validate_every == 0:
                model.eval()
                results = []
                val_losses = []
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        y_hat = model(x)
                        loss = criterion(y_hat, y)
                        y_pred = y_hat.argmax(dim=1)
                        results.append((y == y_pred).float().mean().item())
                        val_losses.append(loss.item())
                run_stats['val_loss'].append(np.mean(val_losses))
                run_stats['val_acc'].append(np.mean(results))
                wandb.log({'val_loss': run_stats['val_loss'][-1], 'val_acc': run_stats['val_acc'][-1]})
                model.train()
            if epoch % args.save_every == 0 and epoch > 0:
                torch.save(model.state_dict(), f'logs/model_{epoch}.pth')
            
            print(f'Epoch {epoch} train loss: {run_stats["train_loss"][-1]} val loss: {run_stats["val_loss"][-1]} val acc: {run_stats["val_acc"][-1]}')
        
        wandb.finish()
