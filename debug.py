"""Debug entrypoint: smoke tests + verification checks, writing outputs/debug."""

from __future__ import annotations

import json
from pathlib import Path

import config
from calibration.runner import build_runner
from calibration.verification import verify_emitter_effect, verify_pda_effect


RUN_SMOKE_TEST = True
RUN_EMITTER_CHECK = True
RUN_PDA_CHECK = True
SAVE_DEBUG_TABLES = True


def _ensure_dirs() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    _ensure_dirs()

    inp_path = config.MODEL_INP
    metadata = config.build_default_metadata()
    runner = build_runner(inp_path=inp_path, metadata=metadata)

    raw = config.build_default_raw_params()

    if RUN_SMOKE_TEST:
        smoke_df = runner.smoke_test(raw)
        if SAVE_DEBUG_TABLES:
            (Path(config.DEBUG_DIR) / "smoke_summary.csv").write_text(smoke_df.to_csv(index=False))
            (Path(config.DEBUG_DIR) / "smoke_debug.json").write_text(
                json.dumps(smoke_df.attrs.get("run_debug", {}), indent=2, default=str)
            )
        print("=== SMOKE TEST ===")
        print(smoke_df.to_string(index=False))
        print("Unmet-demand state:")
        print(smoke_df.attrs.get("unmet_daily_m3", {}))

    if RUN_EMITTER_CHECK:
        off = config.build_default_raw_params()
        off["leakage"]["zone_multipliers"] = {"Z1": 0.0, "Z2": 0.0}

        on = config.build_default_raw_params()
        on["leakage"]["zone_multipliers"] = {"Z1": 40.0, "Z2": 40.0}

        emitter_df = verify_emitter_effect(runner, off, on)
        if SAVE_DEBUG_TABLES:
            (Path(config.DEBUG_DIR) / "emitter_verification.csv").write_text(emitter_df.to_csv(index=False))
            (Path(config.DEBUG_DIR) / "debug_off.json").write_text(
                json.dumps(emitter_df.attrs.get("debug_off", {}), indent=2, default=str)
            )
            (Path(config.DEBUG_DIR) / "debug_on.json").write_text(
                json.dumps(emitter_df.attrs.get("debug_on", {}), indent=2, default=str)
            )
        print("=== EMITTER VERIFICATION (OFF vs STRONG ON) ===")
        print(emitter_df.head().to_string(index=False))

    if RUN_PDA_CHECK:
        low = config.build_default_raw_params()
        low["pda"]["minimum_pressure"] = 1.0
        low["pda"]["required_pressure"] = 10.0

        high = config.build_default_raw_params()
        high["pda"]["minimum_pressure"] = 5.0
        high["pda"]["required_pressure"] = 35.0

        pda_df = verify_pda_effect(runner, low, high)
        if SAVE_DEBUG_TABLES:
            (Path(config.DEBUG_DIR) / "pda_verification.csv").write_text(pda_df.to_csv(index=False))
            (Path(config.DEBUG_DIR) / "debug_low.json").write_text(
                json.dumps(pda_df.attrs.get("debug_low", {}), indent=2, default=str)
            )
            (Path(config.DEBUG_DIR) / "debug_high.json").write_text(
                json.dumps(pda_df.attrs.get("debug_high", {}), indent=2, default=str)
            )
        print("=== PDA VERIFICATION (LOW required pressure vs HIGH required pressure) ===")
        print(pda_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
