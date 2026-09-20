"""Fail-closed comparator for one radiation final-comparison log.

The log format, finite/nonnegative norm checks, and 1e-11 relative final-time
tolerance come from verification_results/radiation/cpp/{data,common}.cpp.
The formerly imported test_problems/radiation/check.py is absent from repository
history, so its purported physics thresholds cannot be recovered. Accordingly,
``check`` requires an explicit complete norm-limit policy; it has no guessed
case-specific defaults. Limits have the same units as the supplied error norms.

This log-only helper does not replace the C++ harness's norm-file, resolution,
slice, conservation, or provenance checks, nor certify a simulation by itself.
"""

from collections.abc import Mapping
import math
import re


FIELDS = ("er", "fx", "fy", "fz")
NORMS = ("L1", "L2", "Linf")
CASES = ("streaming_wave", "streaming_front", "gaussian_pulse", "equilibrium_sphere")
TIME_REL_TOL = 1e-11
_MARKER = re.compile(r"RADIATION_TEST_FINISHED\s+(\w+)\s+t=(\S+)\s*")
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _nonnegative(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid {label}: {value!r}") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Expected finite nonnegative {label}: {value!r}")
    return number


def _log_number(token, label):
    # Accept the decimal/scientific format emitted by the solver, not Python's
    # extra float syntax (e.g. underscores) or non-finite values.
    if not _NUMBER.fullmatch(token):
        raise ValueError(f"Invalid {label}: {token!r}")
    return _nonnegative(token, label)


def parse_norms(text, case, expected_time):
    """Return all twelve norms, rejecting absent, malformed, or stale output.

    Exactly one matching final marker and one row per radiation field are
    required. Known norm rows must follow the marker, so earlier comparison
    values cannot stand in for missing final rows. Unrelated solver log lines
    (including the production ``L1, L2`` header and hydro norms) are permitted.
    Final times use the C++ comparator's purely relative tolerance, including
    exact equality when the expected time is zero.
    """
    if not isinstance(text, str):
        raise ValueError("Expected log text")
    if case not in CASES:
        raise ValueError(f"Unknown radiation case: {case!r}")
    expected_time = _nonnegative(expected_time, "expected time")
    rows = {}
    finished = False
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if line.lstrip().startswith("RADIATION_TEST_FINISHED"):
            marker = _MARKER.fullmatch(line)
            if marker is None:
                raise ValueError("Malformed final comparison marker")
            if finished:
                raise ValueError("Duplicate final comparison marker")
            if marker[1].lower() != "radiation_" + case:
                raise ValueError("Wrong final comparison case")
            time = _log_number(marker[2], "final time")
            if abs(time - expected_time) > TIME_REL_TOL * max(abs(time), abs(expected_time)):
                raise ValueError("Wrong final comparison time")
            finished = True
        elif tokens[0] in FIELDS:
            field = tokens[0]
            if not finished:
                raise ValueError("Norms precede the final comparison marker")
            if field in rows:
                raise ValueError(f"Duplicate norm row: {field}")
            if len(tokens) != 4:
                raise ValueError(f"Expected three norms for {field}")
            rows[field] = {
                norm: _log_number(token, f"{field} {norm}")
                for norm, token in zip(NORMS, tokens[1:])
            }
    if not finished or set(rows) != set(FIELDS):
        raise ValueError("Missing final comparison marker or radiation norms")
    return rows


def check(text, case, expected_time, *, limits):
    """Check the log against explicit inclusive per-field/per-norm limits.

    ``limits`` must contain exactly er/fx/fy/fz, each mapping L1/L2/Linf to a
    finite nonnegative ceiling. No omitted norm, infinite ceiling, or implicit
    physics policy is allowed. Return parsed norms on success; otherwise raise
    ValueError (omitting the required ``limits`` argument raises TypeError).
    """
    if not isinstance(limits, Mapping) or set(limits) != set(FIELDS):
        raise ValueError("An explicit limit policy for all four fields is required")
    validated_limits = {}
    for field in FIELDS:
        if not isinstance(limits[field], Mapping) or set(limits[field]) != set(NORMS):
            raise ValueError(f"An explicit limit for every {field} norm is required")
        validated_limits[field] = {
            norm: _nonnegative(limits[field][norm], f"{field} {norm} limit")
            for norm in NORMS
        }
    rows = parse_norms(text, case, expected_time)
    for field in FIELDS:
        for norm in NORMS:
            value, limit = rows[field][norm], validated_limits[field][norm]
            if value > limit:
                raise ValueError(f"Inaccurate {field} {norm}: {value:.17g} exceeds {limit:.17g}")
    return rows
