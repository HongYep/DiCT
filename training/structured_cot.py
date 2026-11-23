"""Parse structured CoT text and build rollout-token masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


XRAY_MARKER = "[thinking with XRay]"
CT_MARKER = "[thinking with CT volume]"
ANSWER_MARKER = "[Final Answer]"
MARKERS = (XRAY_MARKER, CT_MARKER, ANSWER_MARKER)
TERMINAL_TEXT_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>")


@dataclass(frozen=True)
class StructuredCot:
    xray_reasoning: str
    ct_reasoning: str
    final_block: str
    final_answer: str


def extract_boxed_answer(text: str) -> str:
    marker = r"\boxed{"
    if text.count(marker) != 1:
        return ""
    start = text.index(marker)
    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                suffix = text[index + 1:]
                while suffix:
                    suffix = suffix.lstrip()
                    if not suffix:
                        break
                    terminal = next(
                        (token for token in TERMINAL_TEXT_TOKENS if suffix.startswith(token)), None)
                    if terminal is not None:
                        suffix = suffix[len(terminal):]
                    elif ord(suffix[0]) < 32:
                        suffix = suffix[1:]
                    else:
                        return ""
                return text[content_start:index].strip()
    return ""


def parse_structured_cot(text: Any) -> StructuredCot | None:
    if not isinstance(text, str) or any(text.count(marker) != 1 for marker in MARKERS):
        return None
    xray_start = text.index(XRAY_MARKER) + len(XRAY_MARKER)
    ct_marker = text.index(CT_MARKER)
    ct_start = ct_marker + len(CT_MARKER)
    answer_marker = text.index(ANSWER_MARKER)
    if not (xray_start <= ct_marker < ct_start <= answer_marker):
        return None
    xray = text[xray_start:ct_marker].strip()
    ct = text[ct_start:answer_marker].strip()
    final_block = text[answer_marker + len(ANSWER_MARKER):].strip()
    answer = extract_boxed_answer(final_block)
    if not xray or not ct or not final_block or not answer:
        return None
    return StructuredCot(xray, ct, final_block, answer)


def flatten_token_ids(token_ids: Any) -> tuple[list[int], list[int] | None]:
    """Return flat token IDs and optional per-turn lengths."""
    if not token_ids:
        return [], None
    if isinstance(token_ids[0], int):
        return list(token_ids), None
    lengths = [len(turn) for turn in token_ids]
    return [token for turn in token_ids for token in turn], lengths


def restore_token_shape(values: list[int], lengths: list[int] | None) -> list[int] | list[list[int]]:
    if lengths is None:
        return values
    result = []
    offset = 0
    for length in lengths:
        result.append(values[offset:offset + length])
        offset += length
    return result


def select_masked_token_ids(token_ids: Any, mask: Any) -> list[int] | list[list[int]]:
    """Select sampled token IDs without decoding or re-tokenizing them."""
    flat_ids, id_lengths = flatten_token_ids(token_ids)
    flat_mask, mask_lengths = flatten_token_ids(mask)
    if len(flat_ids) != len(flat_mask) or id_lengths != mask_lengths:
        raise ValueError("Token IDs and routed mask have different shapes")
    if id_lengths is None:
        return [token for token, active in zip(flat_ids, flat_mask) if active]

    result: list[list[int]] = []
    offset = 0
    for length in id_lengths:
        result.append([
            token
            for token, active in zip(flat_ids[offset:offset + length], flat_mask[offset:offset + length])
            if active
        ])
        offset += length
    return result if any(result) else []


def build_structured_token_masks(token_ids: Any, tokenizer: Any) -> dict[str, Any]:
    """Build masks aligned exactly to the sampled response token IDs.

    Valid responses expose the section masks plus a ``non_ct`` mask covering
    every generated token outside the CT-reasoning body. For malformed
    responses, ``non_ct`` covers the whole completion because no CT span can be
    identified reliably.
    """
    flat_ids, turn_lengths = flatten_token_ids(token_ids)
    zero = [0] * len(flat_ids)
    full = [1] * len(flat_ids)

    def invalid_masks() -> dict[str, Any]:
        return {
            "valid": False,
            "xray": restore_token_shape(zero.copy(), turn_lengths),
            "ct": restore_token_shape(zero.copy(), turn_lengths),
            "answer": restore_token_shape(zero.copy(), turn_lengths),
            "control": restore_token_shape(zero.copy(), turn_lengths),
            "non_ct": restore_token_shape(full.copy(), turn_lengths),
            "fallback_grpo": restore_token_shape(full.copy(), turn_lengths),
        }

    text = tokenizer.decode(flat_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != flat_ids:
        return invalid_masks()
    if parse_structured_cot(text) is None:
        return invalid_masks()

    xray_pos = text.index(XRAY_MARKER)
    ct_pos = text.index(CT_MARKER)
    answer_pos = text.index(ANSWER_MARKER)
    spans = {
        "xray": (xray_pos + len(XRAY_MARKER), ct_pos),
        "ct": (ct_pos + len(CT_MARKER), answer_pos),
        "answer": (answer_pos + len(ANSWER_MARKER), len(text)),
    }

    xray_mask = zero.copy()
    ct_mask = zero.copy()
    answer_mask = zero.copy()
    control_mask = zero.copy()
    named_masks = {"xray": xray_mask, "ct": ct_mask, "answer": answer_mask}
    for index, (start, end) in enumerate(encoded["offset_mapping"]):
        for name, (span_start, span_end) in spans.items():
            if start >= span_start and end <= span_end and end > start:
                named_masks[name][index] = 1
    valid = any(xray_mask) and any(ct_mask) and any(answer_mask)
    if not valid:
        return invalid_masks()

    # The marker tokens and any offsets crossing a section boundary are control
    # tokens. Keep generated terminal special tokens out of the answer span too.
    for index in range(len(flat_ids)):
        if not (xray_mask[index] or ct_mask[index] or answer_mask[index]):
            control_mask[index] = 1

    terminal_ids = {
        token_id
        for token_id in (
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
        )
        if token_id is not None
    }
    terminal_start = len(flat_ids)
    while terminal_start > 0 and flat_ids[terminal_start - 1] in terminal_ids:
        terminal_start -= 1
    for index in range(terminal_start, len(flat_ids)):
        xray_mask[index] = 0
        ct_mask[index] = 0
        answer_mask[index] = 0
        control_mask[index] = 1

    non_ct_mask = [int(not active) for active in ct_mask]

    return {
        "valid": valid,
        "xray": restore_token_shape(xray_mask, turn_lengths),
        "ct": restore_token_shape(ct_mask, turn_lengths),
        "answer": restore_token_shape(answer_mask, turn_lengths),
        "control": restore_token_shape(control_mask, turn_lengths),
        "non_ct": restore_token_shape(non_ct_mask, turn_lengths),
        "fallback_grpo": restore_token_shape(zero.copy(), turn_lengths),
    }


def normalize_text(value: Any) -> str:
    return "".join(character.lower() for character in str(value or "") if character.isalnum())


def first_choice(value: Any) -> str:
    text = str(value or "").strip().upper()
    for prefix in ("\\BOXED{", "BOXED{"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    for character in text:
        if character in "ABCD":
            return character
        if character.isalnum():
            break
    return ""


def masks_overlap(*masks: Iterable[int]) -> bool:
    rows = [list(mask) for mask in masks]
    return bool(rows) and any(sum(values) > 1 for values in zip(*rows))
