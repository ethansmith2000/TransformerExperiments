import os
import threading
from types import MethodType
from typing import Dict, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
# from tabulate import tabulate
from transformers import GPT2Config

# Thread-local storage for the active context
_thread_local_storage = threading.local()
_thread_local_storage.active_context = None


class MoeLoadBalancingContext:
    """
    A context manager to collect MoE router logits implicitly during a forward pass.

    Usage:

        # MoeRouter instances internally call MoeLoadBalancingContext.add_logits(...)
        # passing group_name, logits, n_experts, and top_k.

        with MoeLoadBalancingContext(padding_mask) as moe_ctx: # No n_experts or top_k needed here
            # Model forward pass that includes MoeRouter layers
            outputs = model(inputs)

        # Retrieve collected logits and auxiliary loss after the forward pass
        all_router_data = moe_ctx.get_logits_data() # Dict[str, List[Tuple[Tensor, int, int]]]
        aux_loss = moe_ctx.get_aux_loss("mean") # torch.Tensor

        # summarise usage
        moe_ctx.summarise_usage() # print usage summary for all groups
        moe_ctx.summarise_usage(group_name="mlp") # print usage summary for mlp


    """

    def __init__(self, padding_mask: torch.Tensor | None = None):
        """
        Initializes the context manager.

        Args:
            padding_mask (torch.Tensor | None, optional): A boolean mask indicating padding tokens.
                                                        Shape should broadcast to (batch_size * sequence_length).
                                                        Defaults to None.
        """
        self.padding_mask = padding_mask
        self._previous_context = None
        self.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        # Stores tuples of (logits, n_experts, top_k) for each group
        self.logits_data: Dict[str, List[tuple[torch.Tensor, int, int]]] = {}

    def __enter__(self):
        # Store the previous context if nested contexts are used (optional, good practice)
        self._previous_context = getattr(_thread_local_storage, "active_context", None)
        # Set this instance as the active context for the current thread
        _thread_local_storage.active_context = self
        # Clear any data from previous uses of this specific instance
        self.logits_data = {}
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the previous context (if any)
        _thread_local_storage.active_context = self._previous_context
        # No special exception handling needed here, 'with' statement handles propagation

    @classmethod
    def add_logits(cls, group_name: str, logits: torch.Tensor, n_experts: int, top_k: int):
        """Adds router logits and metadata to the currently active MoeLoadBalancingContext."""
        active_context: MoeLoadBalancingContext | None = getattr(_thread_local_storage, "active_context", None)
        if active_context is None:
            return
        if group_name not in active_context.logits_data:
            active_context.logits_data[group_name] = []
        active_context.logits_data[group_name].append((logits, n_experts, top_k))

    def _loss_from_logits(self, logits: torch.Tensor, n_experts: int) -> torch.Tensor:
        # logits shape: (num_tokens, num_experts) where num_tokens = batch_size * sequence_length
        num_tokens = logits.shape[0]
        mask = None  # Initialize mask as None

        # Check if a padding mask exists in the context
        if self.padding_mask is not None:
            flat_mask = self.padding_mask.view(-1)
            # Check if the flattened mask shape matches the number of tokens in logits
            if flat_mask.shape[0] == num_tokens:
                # Ensure mask is boolean and on the correct device
                mask = flat_mask.to(device=logits.device, dtype=torch.bool)
            else:
                raise ValueError(
                    f"Padding mask shape {self.padding_mask.shape} doesn't align with logits shape {logits.shape}."
                )

        gates = F.softmax(logits, dim=1)
        indices1_s = torch.argmax(gates, dim=1)
        mask1 = F.one_hot(indices1_s, num_classes=n_experts)

        # Apply mask if it's valid and exists
        if mask is not None:
            # Select gates and one-hot vectors for non-padded tokens
            masked_gates = gates[mask]
            masked_mask1 = mask1[mask]

            # If all tokens are padded, the loss is 0
            if masked_gates.shape[0] == 0:
                return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

            # Calculate mean over non-padded tokens only
            me = torch.mean(masked_gates, dim=0)
            ce = torch.mean(masked_mask1.float(), dim=0)
        else:
            # Original calculation if no mask or mask mismatch
            me = torch.mean(gates, dim=0)
            ce = torch.mean(mask1.float(), dim=0)

        # Calculate final auxiliary loss
        l_aux = torch.sum(me * ce) * n_experts
        return l_aux

    def get_aux_loss(self, reduce_type: Literal["mean", "sum"] = "mean") -> torch.Tensor:
        total_loss = torch.tensor(
            0.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )  # Initialize on a default device
        num_groups_processed = 0

        for group_name, data_list in self.logits_data.items():
            if not data_list:
                if self.rank == 0:
                    print(
                        f"{self.__class__.__name__}: No logits data for MoE group {group_name}. Skipping loss calculation."
                    )
                continue

            group_losses = []
            first_entry = True
            device = None
            for logits, n_experts, _ in data_list:
                if first_entry:
                    device = logits.device  # Get device from the first tensor
                    total_loss = total_loss.to(device)  # Ensure total_loss is on the correct device
                    first_entry = False
                elif logits.device != device:
                    raise RuntimeError(
                        f"Logits for group {group_name} are on different devices."
                    )  # Or handle device mismatch

                group_losses.append(
                    self._loss_from_logits(logits.to(device), n_experts)
                )  # Ensure logits are on the target device

            if group_losses:
                stacked_losses = torch.stack(group_losses)
                if reduce_type == "mean":
                    total_loss += stacked_losses.mean()
                elif reduce_type == "sum":
                    total_loss += stacked_losses.sum()
                else:
                    raise ValueError(f"Invalid reduce type: {reduce_type}")
                num_groups_processed += 1

        # If reducing across groups (e.g., averaging the mean loss of each group)
        if reduce_type == "mean" and num_groups_processed > 0:
            # If you want the average loss *per group*, divide by the number of groups that had data.
            # If you want the average loss *per layer* across all groups, this isn't quite right,
            # as groups might have different numbers of layers. The current implementation adds the
            # mean loss of each group together. Decide if averaging across groups is needed.
            # For now, let's keep it as sum of means. If average across groups is needed:
            # total_loss /= num_groups_processed
            pass  # Keeping sum of means for now.

        return total_loss

    def get_logits_data(self) -> Dict[str, List[tuple[torch.Tensor, int, int]]]:
        """Returns the collected logits data as a dictionary."""
        return self.logits_data

    @torch.no_grad()
    def summarise_usage(self, group_name: str | None = None):
        target_groups = self.logits_data.keys() if group_name is None else [group_name]

        for k in target_groups:
            if k not in self.logits_data:
                if self.rank == 0:
                    print(f"No data found for MoE group {k}. Skipping usage summary.")
                continue

            data_list = self.logits_data[k]
            if not data_list:
                if self.rank == 0:
                    print(f"No logits data for MoE group {k}. Skipping usage summary.")
                continue

            # Assuming n_experts and top_k are consistent within a group
            _, n_experts, top_k = data_list[0]
            if self.rank == 0:
                print(f"\nSummarising usage for {k} (num_experts={n_experts}, top_k={top_k})")
            self._summarise_usage(data_list)  # Pass the list of tuples

    def _summarise_usage(self, data_list: List[tuple[torch.Tensor, int, int]]):
        # Assume consistency and get n_experts, top_k from the first entry
        if not data_list:
            return
        logits_example, n_experts, top_k = data_list[0]
        device = logits_example.device
        is_rank_0 = self.rank == 0

        # Process padding mask if provided
        unpadded_mask = None
        if self.padding_mask is not None:
            # Use the shape from the first logit tensor in the list for comparison
            flat_padding_mask = self.padding_mask.view(-1).to(device=device, dtype=torch.bool)
            if flat_padding_mask.shape[0] == logits_example.shape[0]:
                unpadded_mask = flat_padding_mask
            elif is_rank_0:
                print(
                    f"Warning: Padding mask shape {self.padding_mask.shape} doesn't align with logits shape {logits_example.shape}. Ignoring mask."
                )

        # Process statistics for each layer
        per_layer_stats = []
        # n_experts is now derived from the data_list
        overall_counts = torch.zeros(n_experts, device=device, dtype=torch.float)  # Use float for counts
        overall_total = 0

        for layer_idx, (logits, _, _) in enumerate(data_list):  # Extract only logits here
            logits = logits.to(device)
            current_tokens = logits.shape[0]

            # Handle layer-specific mask
            layer_mask = None
            layer_token_count = float(current_tokens)  # Use float
            if unpadded_mask is not None:
                # Check mask shape against *current* logits shape
                if unpadded_mask.shape[0] == current_tokens:
                    layer_mask = unpadded_mask
                    layer_token_count = layer_mask.sum().item()
                elif is_rank_0 and layer_idx == 0:  # Only warn once per group if mask mismatch persists
                    print(
                        f"Warning: Mask length ({unpadded_mask.shape[0]}) mismatch for layer {layer_idx} (logits shape {logits.shape}). Ignoring mask for this layer."
                    )

            # Get expert assignments and count
            routing_weights = F.softmax(logits, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)  # Use top_k derived from data_list
            one_hot = (
                F.one_hot(selected_experts, num_classes=n_experts).sum(dim=1).float()
            )  # Sum over top_k dim -> (tokens, n_experts)

            # Count with or without mask
            if layer_mask is not None:
                expert_counts = (one_hot * layer_mask.unsqueeze(-1).float()).sum(dim=0)  # Sum over token dimension
                total_layer = layer_token_count  # Total selections = number of unpadded tokens * top_k
            else:
                expert_counts = one_hot.sum(dim=0)  # Sum over token dimension
                total_layer = float(current_tokens)  # Total selections = number of tokens * top_k

            # Calculate percentages, avoiding division by zero
            # Ensure total_layer is float before division check
            layer_pct = (
                torch.zeros_like(expert_counts) if total_layer < 1e-6 else expert_counts / (total_layer * top_k) * 100
            )  # Divide by total *selections*

            overall_total += total_layer * top_k  # Accumulate total selections
            overall_counts += expert_counts
            per_layer_stats.append((expert_counts, layer_pct, total_layer * top_k))

        # Calculate overall percentages
        overall_pct = torch.zeros_like(overall_counts) if overall_total < 1e-6 else overall_counts / overall_total * 100

        # Only print on rank 0
        if is_rank_0:
            # print(f"\nExpert Usage Summary (num_experts={n_experts}, top_k={top_k}):") # Moved print outside loop

            # Create table
            headers = ["Layer"] + [f"Expert {i}" for i in range(n_experts)] + ["Total Selections"]
            table_data = []

            # Add layer rows
            for layer_idx, (counts, pct, total) in enumerate(per_layer_stats):
                row = [f"Layer {layer_idx}"]
                row.extend([f"{counts[i].item():.0f} ({pct[i].item():.1f}%)" for i in range(n_experts)])
                row.append(f"{total:.0f}")
                table_data.append(row)

            # Add summary row
            summary_row = ["Overall"]
            summary_row.extend(
                [f"{overall_counts[i].item():.0f} ({overall_pct[i].item():.1f}%)" for i in range(n_experts)]
            )
            summary_row.append(f"{overall_total:.0f}")
            table_data.append(summary_row)

            # print(tabulate(table_data, headers=headers, tablefmt="grid"))

            # Print metrics if we have data
            if overall_total > 0:
                imbalance = torch.std(overall_pct).item()
                max_usage = torch.max(overall_pct).item()
                min_usage = torch.min(overall_pct).item()
                mean_pct = torch.mean(overall_pct).item()

                print(f"\nOverall Load Imbalance (std dev): {imbalance:.2f}%")
                print(f"Usage Range: {min_usage:.2f}% - {max_usage:.2f}%")

                if mean_pct > 1e-6:
                    cv = imbalance / mean_pct * 100
                    print(f"Coefficient of Variation: {cv:.2f}%")
                else:
                    print("Coefficient of Variation: N/A (mean usage is near zero)")
            else:
                print("\nNo expert selections recorded. Skipping additional metrics.")

        return overall_counts, overall_pct


### shared expert
class MoeRouter(nn.Module):
    def __init__(
        self,
        group_name: str,
        hidden_size: int,
        expert: List[nn.Module],
        shared_expert: nn.Module | None,
        top_k: int,
        norm_topk_prob: bool,
        output_size: int = None,
    ):
        super().__init__()
        self.group_name = group_name
        self.num_experts = len(expert)
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.output_size = output_size if output_size is not None else hidden_size
        # gating
        self.gate = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(expert)
        self.shared_expert = shared_expert

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)  # <-- RESHAPE HERE
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states_reshaped)  # <-- USE RESHAPED TENSOR

        # --- Add logits to context ---
        # Pass n_experts and top_k along with logits
        MoeLoadBalancingContext.add_logits(self.group_name, router_logits, self.num_experts, self.top_k)
        # ---------------------------

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # routing_weights/selected_experts shape: (batch_size * sequence_length, top_k)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.output_size), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # selected_experts shape: (batch*seq_len, top_k)
        # one_hot output shape: (batch*seq_len, top_k, num_experts) -> 3D
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # expert_mask shape: (num_experts, top_k, batch*seq_len) -> 3D, permute is now valid

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx] shape: (top_k, batch*seq_len)
            idx, top_x = torch.where(expert_mask[expert_idx])
            # idx: index in top_k dimension
            # top_x: index in batch*seq_len dimension

            if top_x.numel() == 0:  # Efficient check if any tokens routed
                continue

            # Index the correct hidden states (use the reshaped version)
            current_state = hidden_states_reshaped[top_x]  # <-- USE RESHAPED and simplify indexing

            # Compute expert output & scale by weight
            # routing_weights shape: (batch*seq_len, top_k)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # Add the contribution
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # Reshape back to original shape
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.output_size)

        if self.shared_expert is not None:
            final_hidden_states = final_hidden_states + self.shared_expert(identity)

        return final_hidden_states


def mlp2moe(
    group_name: str,
    shared_expert: nn.Module,
    n_experts: int,
    top_k: int,
    norm_topk_prob: bool,
    expert_constructor: type[nn.Module] | None = None,
    expert_config: dict | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    use_shared_expert: bool = True,
):
    if expert_constructor is None:
        expert_constructor = shared_expert.__class__
    if expert_config is None:
        expert_config = shared_expert.config
    if intermediate_size is not None:
        expert_config.intermediate_size = intermediate_size
    moe_mlp = MoeRouter(
        group_name=group_name,
        hidden_size=hidden_size,
        expert=[expert_constructor(intermediate_size,expert_config) for _ in range(n_experts)],
        shared_expert=shared_expert if use_shared_expert else None,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
    )
    return moe_mlp


def linear2moe(
    group_name: str,
    shared_expert: nn.Linear,
    n_experts: int,
    top_k: int,
    norm_topk_prob: bool,
    use_shared_expert: bool = True,
):
    """
    Converts a standard nn.Linear layer into a Mixture-of-Experts (MoE) layer
    using MoeRouter, where the original linear layer acts as the shared expert.

    Args:
        shared_expert (nn.Linear): The original linear layer to be used as the shared expert.
        n_experts (int): The number of new experts to create.
        top_k (int): The number of experts to route to for each token.
        norm_topk_prob (bool): Whether to normalize the top-k probabilities.

    Returns:
        MoeRouter: An MoE layer wrapping the new experts and the shared expert.
    """
    # assert isinstance(shared_expert, nn.Linear), "Linear layer expected"
    experts = [
        nn.Linear(shared_expert.in_features, shared_expert.out_features, bias=shared_expert.bias is not None)
        for _ in range(n_experts)
    ]
    moe_layer = MoeRouter(
        group_name=group_name,
        hidden_size=shared_expert.in_features,  # Input dimension for the gate
        expert=experts,
        shared_expert=shared_expert if use_shared_expert else None,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
        output_size=shared_expert.out_features,
    )
    return moe_layer


# lora is a special name for our parameter selection, so the name is funny
def patch_lora_fn(layer, rank=32, alpha=None):
    layer.orig_forward = layer.forward

    def new_forward(self, x):
        orig_outs = self.orig_forward(x)
        resid = self.expert_lor_up(self.expert_lor_down(x))
        return orig_outs + resid * self.expert_lor_scale

    layer.register_module("expert_lor_down", nn.Linear(layer.in_features, rank, bias=False))
    layer.register_module("expert_lor_up", nn.Linear(rank, layer.out_features, bias=False))
    torch.nn.init.kaiming_uniform_(layer.expert_lor_down.weight, a=5**0.5)
    torch.nn.init.zeros_(layer.expert_lor_up.weight)

    alpha = rank if alpha is None or alpha == 0 else alpha
    layer.register_buffer("expert_lor_alpha", torch.tensor(alpha))
    layer.register_buffer("expert_lor_scale", torch.tensor(alpha / rank))
    layer.forward = MethodType(new_forward, layer)


def linear2moe_lora(
    group_name: str,
    linear_layer: nn.Linear,
    n_experts: int,
    top_k: int,
    norm_topk_prob: bool,
    use_shared_expert: bool = True,
):
    """
    Converts a standard nn.Linear layer into a Mixture-of-Experts (MoE) layer
    using MoeRouter, where the original linear layer acts as the shared expert.

    Args:
        shared_expert (nn.Linear): The original linear layer to be used as the shared expert.
        n_experts (int): The number of new experts to create.
        top_k (int): The number of experts to route to for each token.
        norm_topk_prob (bool): Whether to normalize the top-k probabilities.

    Returns:
        MoeRouter: An MoE layer wrapping the new experts and the shared expert.
    """

    patch_lora_fn(linear_layer)

    experts = [
        nn.Linear(
            linear_layer.expert_lor_down.in_features,
            linear_layer.expert_lor_down.out_features,
            bias=linear_layer.expert_lor_down.bias is not None,
        )
        for _ in range(n_experts)
    ]
    moe_layer = MoeRouter(
        group_name=group_name,
        hidden_size=linear_layer.expert_lor_down.in_features,  # Input dimension for the gate
        expert=experts,
        shared_expert=linear_layer.expert_lor_down if use_shared_expert else None,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
        output_size=linear_layer.expert_lor_down.out_features,
    )

    linear_layer.expert_lor_down = moe_layer

    experts = [
        nn.Linear(
            linear_layer.expert_lor_up.in_features,
            linear_layer.expert_lor_up.out_features,
            bias=linear_layer.expert_lor_up.bias is not None,
        )
        for _ in range(n_experts)
    ]
    moe_layer = MoeRouter(
        group_name=group_name,
        hidden_size=linear_layer.expert_lor_up.in_features,  # Input dimension for the gate
        expert=experts,
        shared_expert=linear_layer.expert_lor_up if use_shared_expert else None,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
        output_size=linear_layer.expert_lor_up.out_features,
    )
    linear_layer.expert_lor_up = moe_layer


def attn2moe(
    group_name: str,
    attention_module: nn.Module,  # Expects a module like Gemma3Attention
    n_experts: int,
    top_k: int,
    norm_topk_prob: bool,
    module_keys: List[str] = ["k_proj", "v_proj", "q_proj", "o_proj"],
    use_shared_expert: bool = True,
):
    """
    Converts the Q, K, V projection layers within an attention module to MoE layers in-place.

    Args:
        attention_module (nn.Module): The attention module (e.g., Gemma3Attention) containing
                                      q_proj, k_proj, v_proj linear layers.
        n_experts (int): The number of new experts to create for each projection.
        top_k (int): The number of experts to route to for each token.
        norm_topk_prob (bool): Whether to normalize the top-k probabilities.
    """
    for k in module_keys:
        if k is None:
            continue
        setattr(
            attention_module,
            k,
            linear2moe(
                group_name=group_name,
                shared_expert=getattr(attention_module, k),
                n_experts=n_experts,
                top_k=top_k,
                norm_topk_prob=norm_topk_prob,
                use_shared_expert=use_shared_expert,
            ),
        )
    return attention_module
