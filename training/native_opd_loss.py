"""One OPD objective for aligned CT+answer content, plus XRay reference KL."""

import torch


def opdvr_gate_delta(delta, correctness, closed_mask):
    """Apply the OPDVR ReLU gate to closed-question token log-ratios.

    ``delta`` is teacher_logp - student_logp. Correct closed trajectories keep
    only positive teacher advantages; incorrect ones keep only negative
    teacher advantages. Open-question rows remain vanilla OPD.
    """
    if correctness is None or closed_mask is None:
        raise ValueError('OPDVR requires both trajectory correctness and closed-question masks')
    if correctness.ndim != 1 or closed_mask.ndim != 1 or correctness.shape != closed_mask.shape:
        raise ValueError('OPDVR correctness and closed masks must be matching [batch] vectors')
    if delta.ndim < 1 or delta.shape[0] != correctness.shape[0]:
        raise ValueError('OPDVR batch dimension differs from token log-ratio')
    shape = (correctness.shape[0],) + (1,) * (delta.ndim - 1)
    correct = correctness.to(device=delta.device, dtype=torch.bool).reshape(shape)
    closed = closed_mask.to(device=delta.device, dtype=torch.bool).reshape(shape)
    gated = torch.where(correct, delta.clamp_min(0), delta.clamp_max(0))
    return torch.where(closed, gated, delta)


def xray_reference_kl_loss(student_logps, reference_logps, completion_mask, xray_mask):
    safe_ratio = (reference_logps - student_logps).clamp(-20, 20)
    per_token = (safe_ratio.exp() - safe_ratio - 1).clamp(-10, 10)
    denominator = completion_mask.sum(-1).clamp(min=1)
    return ((per_token * xray_mask * completion_mask).sum(-1) / denominator).mean()


def xray_grpo_loss(student_logps, old_logps, advantages, completion_mask, xray_mask,
                   epsilon_low=0.2, epsilon_high=0.2):
    """Sequence answer advantages applied to the caller-selected token mask.

    The historical name/argument are retained for compatibility. Full-response
    GRPO passes completion_mask as xray_mask; the default caller uses XRay body.
    Match the existing GRPO per-response reduction over full completion length.
    The reference KL is added separately by the caller, never counted twice.
    """
    old = student_logps.detach() if old_logps is None else old_logps.detach()
    ratio = torch.exp(student_logps - old)
    clipped = ratio.clamp(1 - epsilon_low, 1 + epsilon_high)
    advantage = advantages.detach()
    if advantage.ndim == 1:
        advantage = advantage[:, None]
    policy_loss = -torch.minimum(ratio * advantage, clipped * advantage)
    mask = xray_mask & completion_mask
    denominator = completion_mask.sum(-1).clamp(min=1)
    return ((policy_loss * mask).sum(-1) / denominator).mean()


def native_opd_losses(student_logps, old_logps, teacher_logps, reference_logps,
                      completion_mask, teacher_opd_mask, xray_ref_kl_mask, groups,
                      teacher_coefficient=0.5, epsilon_low=0.2, epsilon_high=0.2,
                      teacher_token_weights=None, opdvr_correctness=None,
                      opdvr_closed_mask=None):
    # teacher_logps contains the teacher block log-likelihood divided by the
    # number of student tokens in that block. For identical boundaries a block
    # is one token and this is the ordinary sampled-token OPD advantage.
    old_logps = student_logps.detach() if old_logps is None else old_logps
    flat_positions, group_ids = [], []
    group_count = 0
    for row, row_groups in enumerate(groups):
        for positions in row_groups:
            flat_positions.extend(row * student_logps.shape[1] + p for p in positions)
            group_ids.extend([group_count] * len(positions))
            group_count += 1
    positions = torch.tensor(flat_positions, dtype=torch.long, device=student_logps.device)
    indices = torch.tensor(group_ids, dtype=torch.long, device=student_logps.device)
    old = old_logps.flatten()[positions]
    current = student_logps.flatten()[positions]
    counts = torch.bincount(indices, minlength=group_count).clamp(min=1)
    old_means = old.new_zeros(group_count).scatter_add(0, indices, old) / counts
    # With different tokenizations, use the probability of the same indivisible
    # text block. Vectorize over groups to avoid thousands of tiny GPU kernels.
    group_ratios = current.new_zeros(group_count).scatter_add(0, indices, current - old)
    delta = torch.zeros_like(student_logps).flatten().scatter(
        0, positions, teacher_logps.flatten()[positions] - old_means[indices]).view_as(student_logps)
    log_ratios = torch.zeros_like(student_logps).flatten().scatter(
        0, positions, group_ratios[indices]).view_as(student_logps)
    advantage_delta = delta.detach()
    if opdvr_correctness is not None or opdvr_closed_mask is not None:
        advantage_delta = opdvr_gate_delta(
            advantage_delta, opdvr_correctness, opdvr_closed_mask)
    advantage = teacher_coefficient * advantage_delta
    ratio = torch.exp(log_ratios)
    clipped = ratio.clamp(1 - epsilon_low, 1 + epsilon_high)
    opd_per_token = -torch.minimum(ratio * advantage, clipped * advantage)
    opd_per_token = opd_per_token * teacher_opd_mask
    if teacher_token_weights is not None:
        if teacher_token_weights.shape != student_logps.shape:
            raise ValueError('OPD token weights must share the completion token frame')
        # Preserve clipping and the original per-response denominator. This
        # increases selected answer-token gradients without weakening CT tokens.
        opd_per_token = opd_per_token * teacher_token_weights
    # Keep the existing GRPO reducer (per-response mean, then batch mean).
    # The SFT anchor below has its own selected-token CE normalization.
    denominator = completion_mask.sum(-1).clamp(min=1)
    loss_teacher_opd = ((opd_per_token * completion_mask).sum(-1) / denominator).mean()
    loss_xray_ref_kl = xray_reference_kl_loss(
        student_logps, reference_logps, completion_mask, xray_ref_kl_mask)
    # Preserve a zero gradient path even for an all-invalid local micro-batch.
    loss_teacher_opd = loss_teacher_opd + student_logps.sum() * 0
    return loss_teacher_opd, loss_xray_ref_kl, delta
