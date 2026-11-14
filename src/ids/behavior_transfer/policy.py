"""Frozen strict rank rules and equal component allocations for R25."""

from fractions import Fraction

import numpy as np

from ids.route_batch.metrics import decisions, scores
from ids.route_batch.metrics import threshold as _threshold


def threshold(values, alpha=0.01):
    """Retain the project's finite-sample strict rank and indivisible score ties."""
    return _threshold(values, alpha)


def register(calibration, components, alpha=0.01):
    """Register an OR of fixed components with an equal allocation of the total rate."""
    names = tuple(components)
    if not names or len(set(names)) != len(names) or not 0 < Fraction(str(alpha)) < 1:
        raise ValueError("Expected distinct components and a rate strictly between zero and one")
    share = Fraction(str(alpha)) / len(names)
    # Pass the rational directly so thirds do not lose a rank through float rounding.
    rules = {name: threshold(calibration[name], share) for name in names}
    flags = apply(calibration, rules)
    allowance = len(flags) * Fraction(str(alpha))
    if int(flags.sum()) > allowance.numerator // allowance.denominator:
        raise ValueError("Component registration exceeded the pooled empirical allocation")
    return rules


def apply(frame, rules):
    """Compute the component OR, preserving the order of the supplied rows."""
    if not rules:
        raise ValueError("At least one registered component is required")
    result = None
    for name, rule in rules.items():
        flag = decisions(frame[name], rule)
        if result is not None and flag.shape != result.shape:
            raise ValueError("Decision components have different lengths")
        result = flag if result is None else result | flag
    return result


def prefix_rules(frame, old_rules, components, *, cutoff_ms, alpha=0.01):
    """Freeze the complete first-hour prefix only once its end has been observed.

    The caller applies old_rules for clock < cutoff_ms, and these rules afterward.
    A full frame is accepted so prefix/suffix provenance is recorded in one place.
    """
    names = tuple(components)
    if set(old_rules) != set(names):
        raise ValueError("Prefix and fallback components differ")
    clock = scores(frame["flow_end_ms"])
    if not np.array_equal(clock, np.floor(clock)) or (np.diff(clock) < 0).any():
        raise ValueError("Expected a nondecreasing integer-millisecond log clock")
    prefix = clock < cutoff_ms
    complete = bool(len(clock) and clock[-1] >= cutoff_ms)
    available = complete and bool(prefix.any())
    rules = register(frame.loc[prefix], names, alpha) if available else old_rules
    if available:
        for rule in rules.values():
            rule["scope"] = "local_unlabeled_alert_rate"
            rule["n_reference_rows"] = rule["n_normal"]
    return {
        "rules": rules,
        "cutoff_ms": int(cutoff_ms),
        "local_rows": int(prefix.sum()),
        "status": "local_unlabeled_prefix" if available else "old_calibration_fallback",
        "prefix_complete": complete,
        "fallback_rows": int(prefix.sum()) if available else len(clock),
        "prefix_interpretation": "All observed rows before cutoff, labels unused; rate is an alert-rate allocation on this prefix.",
    }
