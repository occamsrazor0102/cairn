"""Shared helpers for detector modules.

Kept module-private (`_utils`) because nothing outside `beet.detectors`
should depend on these.
"""
from __future__ import annotations


def interpolate(x: float, table: list[tuple[float, float]]) -> float:
    """Piecewise-linear interpolation over a sorted (x, y) table.

    `table` must be sorted by the first element. Out-of-range x values
    clamp to the nearest endpoint y.
    """
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        x0, y0 = table[i]
        x1, y1 = table[i + 1]
        if x0 <= x <= x1:
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return table[-1][1]
