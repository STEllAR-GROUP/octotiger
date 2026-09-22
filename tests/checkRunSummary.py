#!/usr/bin/env python3
"""Compare named numerical fields in an Octo-Tiger run-summary JSON file."""

import argparse
import json
import math
from pathlib import Path


def fieldValue(document, fieldPath):
    value = document
    for component in fieldPath.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"Missing result field: {fieldPath}")
        value = value[component]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Result field is not numeric: {fieldPath}")
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--expect", nargs=2, action="append", default=[],
                        metavar=("FIELD", "VALUE"))
    parser.add_argument("--relative-tolerance", dest="relativeTolerance", type=float, default=1.0e-6)
    parser.add_argument("--absolute-tolerance", dest="absoluteTolerance", type=float, default=1.0e-14)
    arguments = parser.parse_args()

    with arguments.summary.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("status") != "completed":
        raise ValueError("Run summary does not describe a completed run")
    for fieldPath, expectedText in arguments.expect:
        actual = fieldValue(document, fieldPath)
        expected = float(expectedText)
        if not math.isfinite(actual) or not math.isclose(actual, expected,
                rel_tol=arguments.relativeTolerance,
                abs_tol=arguments.absoluteTolerance):
            raise ValueError(
                f"{fieldPath}: got {actual:.17g}, expected {expected:.17g}")


if __name__ == "__main__":
    main()
