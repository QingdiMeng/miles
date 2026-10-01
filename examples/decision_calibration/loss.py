"""Unclipped one-step policy-gradient surrogate for the Brier objective."""

from argparse import Namespace
from collections.abc import Callable

import torch

from miles.backends.training_utils.loss_hub.logit_processors import _iter_response_chunks
from miles.backends.training_utils.parallel import get_parallel_state


def action_surrogate(logits: torch.Tensor, old_probabilities: torch.Tensor, target: torch.Tensor, action: int) -> torch.Tensor:
    probabilities = logits.float().softmax(dim=-1)
    # Refresh the calibration reward at learner weights. Importance sampling
    # then gives the current Brier gradient even for stale async actions.
    reward = (2 * (target[action] - probabilities[action])).detach()
    ratio = probabilities[action] / old_probabilities[action].detach()
    return -ratio * reward


def policy_loss(args: Namespace, batch: dict, logits: torch.Tensor, sum_of_sample_mean: Callable) -> tuple:
    parallel = get_parallel_state()
    if parallel.tp.size != 1 or parallel.cp.size != 1:
        raise ValueError("This pilot requires TP=1 and CP=1; expert parallelism is supported")
    if any(length != 1 for length in batch["response_lengths"]):
        raise ValueError("Calibration rollouts must contain exactly one action token")
    chunks = _iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
        include_response_indices=False,
    )
    losses, briers, entropies, gaps = [], [], [], []
    for (chunk, tokens, _), metadata in zip(chunks, batch["metadata"], strict=True):
        candidate_ids = metadata["candidate_token_ids"]
        action = metadata["action"]
        if tokens.numel() != 1 or tokens.item() != candidate_ids[action]:
            raise ValueError("Sample action does not match its training metadata")
        candidate_logits = chunk[0, candidate_ids].float()
        old = torch.as_tensor(metadata["probabilities"], device=logits.device, dtype=torch.float32)
        target = torch.as_tensor(metadata["target"], device=logits.device, dtype=torch.float32)
        probabilities = candidate_logits.softmax(dim=-1)
        losses.append(action_surrogate(candidate_logits, old, target, action))
        briers.append((probabilities - target).square().sum().detach())
        entropies.append(-(probabilities * probabilities.clamp_min(1e-30).log()).sum().detach())
        gaps.append((probabilities.detach() - old).abs().max())
    loss = sum_of_sample_mean(torch.stack(losses))
    return loss, {
        "loss": loss.detach(),
        "brier_excess": sum_of_sample_mean(torch.stack(briers)),
        "decision_entropy": sum_of_sample_mean(torch.stack(entropies)),
        "decision_probability_abs_diff": sum_of_sample_mean(torch.stack(gaps)),
    }
