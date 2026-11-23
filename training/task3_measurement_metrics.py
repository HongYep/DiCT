"""Parse and compare dimensioned measurements for GRPO rewards."""

from __future__ import annotations

import re
from typing import Any

NUMBER = re.compile(r"\d+(?:\.\d+)?")
LENGTH_UNIT = re.compile(r"(mm|millimeters?|cm|centimeters?)\b", re.IGNORECASE)
RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)")
PERCENT = re.compile(r"(?:\\?%|\bpercent\b)", re.IGNORECASE)
LOWER_BOUND = re.compile(r"(?:>=|>|≥|more than|greater than|over|above|at least)", re.IGNORECASE)
UPPER_BOUND = re.compile(r"(?:<=|<|≤|less than|under|below|at most)", re.IGNORECASE)


def parse_measurement(text: Any) -> dict[str, Any] | None:
    value = str(text or "").strip().replace("×", "x").replace("\\times", "x")
    numbers = [float(match.group()) for match in NUMBER.finditer(value)]
    if not numbers:
        return None
    inequality = "lower" if LOWER_BOUND.search(value) else "upper" if UPPER_BOUND.search(value) else None
    if PERCENT.search(value):
        range_match = RANGE.search(value)
        interval = [float(range_match.group(1)), float(range_match.group(2))] if range_match else None
        parsed_values = [sum(interval) / 2] if interval else numbers
        return {"kind": "percent", "values": parsed_values, "interval": interval,
                "inequality": inequality, "unit": "%"}
    unit_matches = list(LENGTH_UNIT.finditer(value))
    if not unit_matches:
        return None
    factor_values = [
        10.0 if match.group().lower().startswith(("cm", "centimeter")) else 1.0
        for match in unit_matches
    ]
    factors = set(factor_values)
    if len(factors) == 1:
        converted = [number * factor_values[0] for number in numbers]
    else:
        number_matches = list(NUMBER.finditer(value))
        converted = []
        for index, (number, match) in enumerate(zip(numbers, number_matches)):
            boundary = number_matches[index + 1].start() if index + 1 < len(number_matches) else len(value)
            local_units = [
                unit for unit in unit_matches if match.end() <= unit.start() < boundary
            ]
            if len(local_units) != 1:
                return None
            unit = local_units[0].group().lower()
            factor = 10.0 if unit.startswith(("cm", "centimeter")) else 1.0
            converted.append(number * factor)
    range_match = RANGE.search(value)
    has_dimension_separator = bool(re.search(r"\d\s*x\s*\d|\bby\b", value, re.IGNORECASE))
    interval = None
    if range_match and len(converted) == 2 and not has_dimension_separator:
        interval = sorted(converted)
        converted = [sum(interval) / 2]
    return {
        "kind": "length",
        "values": converted,
        "interval": interval,
        "inequality": inequality,
        "unit": "mm",
    }


def component_errors(reference: dict[str, Any], prediction: dict[str, Any]) -> tuple[list[float], list[float]]:
    reference_values = sorted(reference["values"], reverse=True)
    prediction_values = sorted(prediction["values"], reverse=True)
    errors: list[float] = []
    relative: list[float] = []
    for target, predicted in zip(reference_values, prediction_values):
        reference_interval = reference.get("interval")
        prediction_interval = prediction.get("interval")
        if reference_interval and prediction_interval:
            error = max(0.0, reference_interval[0] - prediction_interval[1],
                        prediction_interval[0] - reference_interval[1])
        elif reference_interval:
            error = max(0.0, reference_interval[0] - predicted, predicted - reference_interval[1])
        elif prediction_interval:
            error = max(0.0, prediction_interval[0] - target, target - prediction_interval[1])
        elif reference["inequality"] == "upper":
            error = max(0.0, predicted - target)
        elif reference["inequality"] == "lower":
            error = max(0.0, target - predicted)
        else:
            error = abs(predicted - target)
        errors.append(error)
        relative.append(error / max(abs(target), 1e-12))
    return errors, relative
