"""Parse native OPD spans without re-tokenizing the student's sampled trajectory.

Character spans describe text; token spans use UTF-8 bytes. An indivisible token
crossing a semantic/format boundary is excluded from ALL three objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from structured_cot import MARKERS, TERMINAL_TEXT_TOKENS, parse_structured_cot


@dataclass
class ContentSpans:
    text: str
    xray: tuple[int, int] | None
    ct: tuple[int, int] | None
    answer: tuple[int, int] | None
    format_valid: bool


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _escaped(text: str, index: int) -> bool:
    slashes = 0
    while index > 0 and text[index - 1] == "\\":
        slashes += 1
        index -= 1
    return slashes % 2 == 1


def parse_content(text: str) -> ContentSpans:
    """Local extraction: a missing Final Answer header need not erase CT+answer.

    A unique closed box and a unique CT header are required for teacher scoring.
    XRay extraction is independent, so malformed CT/answer cannot disable its KL.
    Header spellings are deliberately the existing SFT spellings in MARKERS.
    """
    body_end = len(text)
    for terminal in TERMINAL_TEXT_TOKENS:
        at = text.find(terminal)
        if at >= 0:
            body_end = min(body_end, at)
    header_positions = [text.find(m) if text.count(m) == 1 else -1 for m in MARKERS]
    xp, cp, fp = header_positions
    xray = ct = answer = None
    box_start = -1
    box_marker = r"\boxed{"
    if text.count(box_marker) == 1:
        box_start = text.index(box_marker)
        start = box_start + len(box_marker)
        depth = 1
        for i in range(start, body_end):
            if not _escaped(text, i):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
            if depth == 0:
                answer = _trim(text, start, i)
                break
    if xp >= 0:
        start = xp + len(MARKERS[0])
        # Even repeated downstream headers give an unambiguous end to XRay.
        ends = [body_end]
        for marker in MARKERS[1:]:
            at = text.find(marker, start)
            if at >= 0:
                ends.append(at)
        if box_start >= start:
            ends.append(box_start)
        xray = _trim(text, start, min(ends))
    if cp >= 0:
        start = cp + len(MARKERS[1])
        # Reject repeated/misordered Final Answer headers, but allow a missing one.
        if text.count(MARKERS[2]) == 0 and box_start >= start:
            ct = _trim(text, start, box_start)
        elif fp >= start and (box_start < 0 or fp < box_start):
            ct = _trim(text, start, fp)
        # A reserved header inside the extracted content makes it ambiguous.
        if ct and any(marker in text[ct[0]:ct[1]] for marker in MARKERS):
            ct = None
        if answer and answer[0] < start:
            answer = None
    return ContentSpans(text, xray, ct, answer, parse_structured_cot(text) is not None)


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    # The reversible byte alphabet used by GPT-2/Qwen ByteLevel tokenizers.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs.copy()
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return {chr(char): byte for byte, char in zip(bs, cs)}


def token_byte_offsets(ids: list[int], tokenizer: Any) -> tuple[str, list[tuple[int, int]]]:
    """Offsets for ORIGINAL IDs, including alternate BPE splits and Unicode bytes.

    Prefer exact reversible byte decoding. Other tokenizer families use decoded
    prefixes; if decoding rewrites an earlier prefix, fail explicitly rather than
    silently apply a mask to different tokens.
    """
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    target = text.encode("utf-8")
    decoder = _byte_decoder()
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    pieces = []
    try:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        for token_id, token in zip(ids, tokens):
            pieces.append(token.encode("utf-8") if token_id in special_ids
                          else bytes(decoder[c] for c in token))
    except (AttributeError, KeyError, TypeError, ValueError):
        pieces = []
    if pieces and b"".join(pieces) == target:
        result, offset = [], 0
        for piece in pieces:
            result.append((offset, offset + len(piece)))
            offset += len(piece)
        return text, result

    # This path also supports small test tokenizers and SentencePiece-like decoders.
    boundaries = [0]
    for end in range(1, len(ids) + 1):
        prefix = tokenizer.decode(ids[:end], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        prefix = prefix.encode("utf-8")
        common = 0
        for a, b in zip(prefix, target):
            if a != b:
                break
            common += 1
        if common < boundaries[-1]:
            raise ValueError("Tokenizer rewrites earlier decoded text; cannot align sampled tokens safely")
        boundaries.append(common)
    if boundaries[-1] != len(target):
        raise ValueError("Could not align decoded completion to original token IDs")
    # A token with zero width (e.g. an unfinished Unicode codepoint) is inseparable
    # from the next token. Give them a common span to form a shared alignment block.
    offsets = list(zip(boundaries[:-1], boundaries[1:]))
    for i, (start, end) in enumerate(offsets):
        if start == end:
            following = next((b for a, b in offsets[i + 1:] if b > start), None)
            if following is None:
                raise ValueError("Unalignable trailing zero-width token")
            offsets[i] = (start, following)
    return text, offsets


def byte_span(text: str, span: tuple[int, int] | None) -> tuple[int, int] | None:
    if span is None:
        return None
    return len(text[:span[0]].encode("utf-8")), len(text[:span[1]].encode("utf-8"))


def span_mask(offsets, span) -> list[int]:
    if span is None:
        return [0] * len(offsets)
    lo, hi = span
    return [int(lo <= a < b <= hi) for a, b in offsets]


def assert_disjoint_masks(*masks):
    if len({len(mask) for mask in masks}) != 1:
        raise ValueError("Supervision masks must share the same token frame")
    if any(sum(bool(value) for value in values) > 1 for values in zip(*masks)):
        raise ValueError("Teacher OPD, XRay reference KL and format anchor masks overlap")


def format_token_mask(text, offsets, spans):
    """Only headers, outer box syntax, structural whitespace and terminal EOS.

    Neither a semantic token nor a mixed semantic/format token is eligible.
    Missing local content spans do not turn arbitrary prose into formatting.
    """
    allowed = bytearray(len(text.encode('utf-8')))

    def mark(start, end):
        a, b = byte_span(text, (start, end))
        allowed[a:b] = b'\x01' * (b - a)

    for marker in MARKERS:
        if text.count(marker) == 1:
            start = text.index(marker)
            mark(start, start + len(marker))
    for token in TERMINAL_TEXT_TOKENS:
        if text.endswith(token) or text.rstrip().endswith(token):
            start = text.rfind(token)
            mark(start, start + len(token))
    if text.count(r'\boxed{') == 1 and spans.answer is not None:
        start = text.index(r'\boxed{')
        content_start = start + len(r'\boxed{')
        mark(start, content_start)
        depth = 1
        for i in range(content_start, len(text)):
            if not _escaped(text, i):
                depth += int(text[i] == '{') - int(text[i] == '}')
            if depth == 0:
                mark(i, i + 1)
                break
    semantic = [span for span in (spans.xray, spans.ct, spans.answer) if span is not None]
    # Separator/newline supervision is defined only when all section boundaries
    # are known. Replay samples are required to satisfy this condition.
    if len(semantic) == 3:
        for i, char in enumerate(text):
            if char.isspace() and not any(a <= i < b for a, b in semantic):
                mark(i, i + 1)
    # A header quoted inside semantic prose must not become a format target.
    for span in semantic:
        a, b = byte_span(text, span)
        allowed[a:b] = b'\x00' * (b - a)
    return [int(a < b and all(allowed[a:b])) for a, b in offsets]


def _alignment_groups(student_offsets, teacher_offsets, student_span, teacher_span):
    """Smallest common byte-boundary blocks; never divide a tokenizer's token."""
    slo, shi = student_span
    tlo, thi = teacher_span
    assert shi - slo == thi - tlo
    student = [(i, a - slo, b - slo) for i, (a, b) in enumerate(student_offsets)
               if slo <= a < b <= shi]
    teacher = [(i, a - tlo, b - tlo) for i, (a, b) in enumerate(teacher_offsets)
               if tlo <= a < b <= thi]

    def boundaries(tokens):
        values = {v for _, a, b in tokens for v in (a, b)}
        # Overlapping byte spans from Unicode fallback are a single indivisible block.
        result, index, furthest = set(), 0, -1
        for value in sorted(values):
            while index < len(tokens) and tokens[index][1] < value:
                furthest = max(furthest, tokens[index][2])
                index += 1
            if furthest <= value:
                result.add(value)
        return result

    shared = sorted(boundaries(student) & boundaries(teacher))
    groups = []
    student_index = teacher_index = 0
    for start, end in zip(shared, shared[1:]):
        st, tt = [], []
        while student_index < len(student) and student[student_index][1] < end:
            item = student[student_index]
            if start <= item[1] < item[2] <= end:
                st.append(item)
            student_index += 1
        while teacher_index < len(teacher) and teacher[teacher_index][1] < end:
            item = teacher[teacher_index]
            if start <= item[1] < item[2] <= end:
                tt.append(item)
            teacher_index += 1
        def covers(tokens):
            at = start
            for _, a, b in tokens:
                if a > at:
                    return False
                at = max(at, b)
            return bool(tokens) and tokens[0][1] == start and at == end
        if covers(st) and covers(tt):
            groups.append({"student": [i for i, _, _ in st], "teacher": [i for i, _, _ in tt]})
    return groups


def build_xray_teacher_plan(student_ids: list[int], student_tokenizer, teacher_tokenizer) -> dict:
    """Score only the sampled XRay body in the image teacher's native context.

    The teacher never sees CT text, boxed answers, or the training-only section
    headers. Retain usable XRay prefixes even when a later section is truncated.
    """
    text, offsets = token_byte_offsets(student_ids, student_tokenizer)
    spans = parse_content(text)
    plan = {'teacher_ids': [], 'teacher_target_text': '', 'groups': [],
            'teacher_opd_mask': [0] * len(student_ids)}
    if spans.xray is None:
        return plan
    native = text[slice(*spans.xray)]
    if any(marker in native for marker in MARKERS):
        return plan
    teacher_ids = teacher_tokenizer.encode(native, add_special_tokens=False)
    decoded, teacher_offsets = token_byte_offsets(teacher_ids, teacher_tokenizer)
    if decoded != native:
        raise ValueError('XRay teacher tokenizer normalized the sampled target')
    groups = _alignment_groups(offsets, teacher_offsets, byte_span(text, spans.xray),
                               (0, len(native.encode('utf-8'))))
    for group in groups:
        for i in group['student']:
            plan['teacher_opd_mask'][i] = 1
    plan.update(teacher_ids=teacher_ids, teacher_target_text=native, groups=groups)
    return plan


def build_native_plan(student_ids: list[int], student_tokenizer, teacher_tokenizer) -> dict:
    text, offsets = token_byte_offsets(student_ids, student_tokenizer)
    spans = parse_content(text)
    xray_mask = span_mask(offsets, byte_span(text, spans.xray))
    ct_mask = span_mask(offsets, byte_span(text, spans.ct))
    answer_mask = span_mask(offsets, byte_span(text, spans.answer))
    formatting = format_token_mask(text, offsets, spans)
    assert_disjoint_masks(ct_mask, answer_mask, xray_mask, formatting)
    plan = {
        "teacher_opd_mask": [0] * len(student_ids), "xray_ref_kl_mask": xray_mask,
        "ct_body_mask": [0] * len(student_ids), "answer_content_mask": [0] * len(student_ids),
        "format_token_mask": formatting,
        "teacher_ids": [], "teacher_target_text": "", "groups": [],
        "format_valid": spans.format_valid, "ct_parse_success": spans.ct is not None,
        "answer_parse_success": spans.answer is not None,
        "xray_length": sum(xray_mask), "ct_length": sum(ct_mask), "answer_length": sum(answer_mask),
    }
    if spans.ct is None or spans.answer is None:
        return plan
    ct = text[slice(*spans.ct)]
    answer = text[slice(*spans.answer)]
    if any(m in ct or m in answer for m in MARKERS):
        return plan
    native = ct + "\n\\boxed{" + answer + "}"
    # Always tokenize with the teacher's tokenizer, even when the vocabularies match.
    teacher_ids = teacher_tokenizer.encode(native, add_special_tokens=False)
    decoded, teacher_offsets = token_byte_offsets(teacher_ids, teacher_tokenizer)
    if decoded != native:
        raise ValueError("Teacher tokenizer normalizes the target text; exact semantic alignment is unavailable")
    teacher_ct = (0, len(ct))
    answer_start = len(ct) + len("\n\\boxed{")
    teacher_answer = (answer_start, answer_start + len(answer))
    groups = []
    for name, ss, ts in [("ct", spans.ct, teacher_ct), ("answer", spans.answer, teacher_answer)]:
        section = _alignment_groups(offsets, teacher_offsets, byte_span(text, ss), byte_span(native, ts))
        for group in section:
            group["section"] = name
        groups.extend(section)
    plan.update(teacher_ids=teacher_ids, teacher_target_text=native, groups=groups)
    for group in groups:
        for i in group["student"]:
            plan['ct_body_mask' if group['section'] == 'ct' else 'answer_content_mask'][i] = 1
    plan['teacher_opd_mask'] = [int(ct or answer) for ct, answer in
                                zip(plan['ct_body_mask'], plan['answer_content_mask'])]
    assert_disjoint_masks(plan['teacher_opd_mask'], xray_mask, formatting)
    return plan


def sft_anchor_mask(ids: list[int], tokenizer) -> list[int]:
    """SFT replay: protocol ONLY, excluding XRay/CT/answer content and mixed tokens."""
    text, offsets = token_byte_offsets(ids, tokenizer)
    spans = parse_content(text)
    if not spans.format_valid:
        raise ValueError("Original SFT anchor must have a valid complete three-section response")
    formatting = format_token_mask(text, offsets, spans)
    content = [span_mask(offsets, byte_span(text, span)) for span in (spans.xray, spans.ct, spans.answer)]
    assert_disjoint_masks(formatting, *content)
    if not any(formatting):
        raise ValueError("SFT replay response has no unambiguous format tokens")
    return formatting
