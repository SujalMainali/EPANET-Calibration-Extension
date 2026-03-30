"""Helpers for working with the WNTR EPANET toolkit wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

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


@dataclass(frozen=True)
class EpanetToolkitError(RuntimeError):
    method: str
    code: int
    message: Optional[str] = None

    def __str__(self) -> str:
        base = f"EPANET toolkit call {self.method} failed with code {self.code}"
        return f"{base}: {self.message}" if self.message else base


def _try_get_error_message(en: Any, code: int) -> Optional[str]:
    """Best-effort error message retrieval across wrapper variants."""
    getter = getattr(en, "ENgeterror", None)
    if getter is None:
        return None

    try:
        out = getter(code)
        if isinstance(out, tuple) and len(out) >= 2:
            return str(out[-1])
        if out is None:
            return None
        return str(out)
    except Exception:
        return None


def call_en_get(en: Any, method_name: str, *args):
    """Call a toolkit function and return its value.

    WNTR's ENepanet wrapper typically returns tuples like (err, value).
    This helper raises on nonzero err codes so failures are never silent.
    """
    method = getattr(en, method_name)
    out = method(*args)

    if isinstance(out, tuple):
        if len(out) == 0:
            return None

        first = out[0]
        if isinstance(first, int) and first != 0:
            raise EpanetToolkitError(method=method_name, code=first, message=_try_get_error_message(en, first))

        if len(out) >= 2:
            return out[-1]
        return None

    return out


def call_en_get_int(en: Any, method_name: str, *args) -> int:
    """Call an EN toolkit getter expected to return an integer-like value."""
    value = call_en_get(en, method_name, *args)
    if value is None:
        raise RuntimeError(f"Toolkit call {method_name} returned no value")
    return int(value)
