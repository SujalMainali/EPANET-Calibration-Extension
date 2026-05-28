#!/usr/bin/env python
"""Validate recalibration parameters to confirm freezes are working."""

import json
from pathlib import Path

def validate_calibration_params():
    """Validate that frozen parameters remained frozen after recalibration."""
    
    best_params_path = Path('outputs/reports/best_params.json')
    if not best_params_path.exists():
        print("❌ best_params.json not found!")
        return False
    
    with open(best_params_path) as f:
        params = json.load(f)
    
    print("\n" + "="*70)
    print("RECALIBRATION PARAMETER VALIDATION")
    print("="*70)
    
    # Check frozen parameters
    print(f"\n🔒 FROZEN PARAMETERS (should be unchanged):")
    
    # Demand multiplier
    dm = params.get('demand', {}).get('demand_multiplier', 'NOT FOUND')
    status = "✅" if dm == 1.0 else "❌"
    print(f"  {status} demand.demand_multiplier = {dm} (expected: 1.0)")
    
    # Duration days
    duration = params.get('time', {}).get('duration_days', 'NOT FOUND')
    status = "✅" if duration == 1 else "❌"
    print(f"  {status} time.duration_days = {duration} (expected: 1)")
    
    # Pattern family parameters (should not exist or be default)
    pf = params.get('pattern_family', {})
    print(f"\n  pattern_family parameters found: {len(pf)} (expected: 0 or minimal)")
    if pf:
        for key, val in pf.items():
            print(f"    - {key}: {val}")
    
    # Check free parameters (should be updated)
    print(f"\n🔓 FREE PARAMETERS (should be optimized):")
    
    pda_params = params.get('pda', {})
    print(f"\n  PDA Parameters:")
    for key, val in pda_params.items():
        print(f"    - {key}: {val}")
    
    leakage = params.get('leakage', {})
    print(f"\n  Leakage Parameters:")
    if isinstance(leakage, dict):
        for key, val in leakage.items():
            if key == 'zone_multipliers':
                print(f"    - zone_multipliers:")
                for zone, mult in val.items():
                    print(f"        {zone}: {mult:.6f}")
            else:
                print(f"    - {key}: {val}")
    
    print("\n" + "="*70)
    if dm == 1.0 and duration == 1:
        print("✅ PARAMETER VALIDATION PASSED")
        print("   Frozen parameters are correct!")
    else:
        print("❌ PARAMETER VALIDATION FAILED")
        print("   Check if recalibration respected frozen parameters!")
    print("="*70 + "\n")
    
    return dm == 1.0 and duration == 1

if __name__ == "__main__":
    validate_calibration_params()
