"""Scalar parser and reward for the ReXVQA measurement training subset."""
import math
import re


def parse_scalar(box):
    text = box.strip().replace('$', '')
    text = re.sub(r'\\(?:text|mathrm|textrm|operatorname)\s*\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\(?:approx|sim)\s*', '', text)
    text = text.replace(r'\,', ' ').replace(r'\;', ' ').replace(r'\!', '').replace('~', ' ')
    match = re.fullmatch(r'\s*(?:(?:approximately|about|approx\.?)[ ]+)?'
                         r'([+]?(?:\d+(?:\.\d*)?|\.\d+))\s*'
                         r'(mm|millimeters?|cm|centimeters?)?\s*\.?\s*', text, re.I)
    if not match:
        return None
    value = float(match[1])
    if match[2] and match[2].lower().startswith(('cm', 'centimeter')):
        value *= 10
    return value if math.isfinite(value) and value >= 0 else None


def extract_boxed_answer(text):
    marker = r'\boxed{'
    start = text.rfind(marker)
    if start < 0:
        return '', False
    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
            if depth == 0:
                return text[content_start:index].strip(), True
    return '', False


def numerical_score(completion, reference_mm, finish_reason=None, is_truncated=False):
    reference_mm = float(reference_mm)
    if not math.isfinite(reference_mm) or reference_mm < 0:
        raise ValueError('A finite nonnegative reference in millimeters is required')
    answer, boxed = extract_boxed_answer(str(completion))
    value = parse_scalar(answer) if boxed else None
    if finish_reason not in (None, '', 'stop') or is_truncated:
        value = None
    error = abs(value - reference_mm) if value is not None else None
    scale = max(5.0, 0.2 * reference_mm)
    reward = math.exp(-error / scale) if error is not None else 0.0
    return {'response': answer, 'boxed_answer_found': boxed, 'predicted_mm': value,
            'reference_mm': reference_mm, 'absolute_error_mm': error,
            'reward_scale_mm': scale, 'reward': reward}
