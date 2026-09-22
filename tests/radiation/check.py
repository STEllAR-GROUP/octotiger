"""Fail-closed comparator for structured radiation run summaries."""

from collections.abc import Mapping
import math

fields = ("er", "fx", "fy", "fz")
norms = ("L1", "L2", "Linf")
normKeys = {"L1": "l1", "L2": "l2", "Linf": "linf"}
cases = ("streaming_wave", "streaming_front", "gaussian_pulse", "equilibrium_sphere")
timeRelTol = 1e-11


def nonnegative(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid {label}: {value!r}") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Expected finite nonnegative {label}: {value!r}")
    return number


def readNorms(summary, case, expectedTime):
    if not isinstance(summary, Mapping):
        raise ValueError("Expected a structured run summary")
    if case not in cases:
        raise ValueError(f"Unknown radiation case: {case!r}")
    expectedTime = nonnegative(expectedTime, "expected time")
    if summary.get("schemaVersion") != 1 or summary.get("status") != "completed":
        raise ValueError("Run summary is missing, incompatible, or incomplete")
    problem = summary.get("problem")
    if problem is not None and str(problem).lower() != "radiation_" + case:
        raise ValueError("Wrong radiation problem")
    finalTime = nonnegative(summary.get("finalTime"), "final time")
    if abs(finalTime - expectedTime) > timeRelTol * max(abs(finalTime), abs(expectedTime)):
        raise ValueError("Wrong final comparison time")
    analytic = summary.get("analytic")
    if not isinstance(analytic, Mapping) or set(analytic) != set(fields):
        raise ValueError("Missing or unexpected radiation analytic fields")
    rows = {}
    for field in fields:
        source = analytic[field]
        if not isinstance(source, Mapping) or set(source) != set(normKeys.values()):
            raise ValueError(f"Missing or unexpected norms for {field}")
        rows[field] = {norm: nonnegative(source[key], f"{field} {norm}")
                       for norm, key in normKeys.items()}
    return rows


def check(summary, case, expectedTime, *, limits):
    if not isinstance(limits, Mapping) or set(limits) != set(fields):
        raise ValueError("An explicit limit policy for all four fields is required")
    validatedLimits = {}
    for field in fields:
        if not isinstance(limits[field], Mapping) or set(limits[field]) != set(norms):
            raise ValueError(f"An explicit limit for every {field} norm is required")
        validatedLimits[field] = {
            norm: nonnegative(limits[field][norm], f"{field} {norm} limit")
            for norm in norms
        }
    rows = readNorms(summary, case, expectedTime)
    for field in fields:
        for norm in norms:
            value, limit = rows[field][norm], validatedLimits[field][norm]
            if value > limit:
                raise ValueError(f"Inaccurate {field} {norm}: {value:.17g} exceeds {limit:.17g}")
    return rows
