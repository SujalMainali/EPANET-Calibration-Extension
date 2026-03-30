"""Helpers for working with the WNTR EPANET toolkit wrappers."""

from __future__ import annotations

from typing import Any

from wntr.epanet.util import EN


def safe_float(x: Any, default=float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def EN_const(name: str) -> int:
    """Resolve an EPANET toolkit constant in a WNTR-version-safe way."""
    if hasattr(EN, name):
        return getattr(EN, name)
    if hasattr(EN, "EN_" + name):
        return getattr(EN, "EN_" + name)
    raise AttributeError(f"Cannot find EPANET constant {name} in wntr.epanet.util.EN")


def call_en_get(en: Any, method_name: str, *args):
    """Call an EN toolkit getter; return last tuple element if tuple."""
    method = getattr(en, method_name)
    out = method(*args)
    if isinstance(out, tuple):
        if len(out) == 0:
            return None
        if len(out) >= 2:
            return out[-1]
    return out


def call_en_get_int(en: Any, method_name: str, *args) -> int:
    """Call an EN toolkit getter expected to return an integer-like value."""
    value = call_en_get(en, method_name, *args)
    if value is None:
        raise RuntimeError(f"Toolkit call {method_name} returned no value")
    return int(value)
