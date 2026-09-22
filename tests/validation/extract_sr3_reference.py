"""
Utility to extract reference values from SR3 files for CMG benchmark cases.
Run this to verify OOIP and other parameters from actual CMG simulation outputs.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from validation.sr3_parser import SR3Parser


SR3_FILES = {
    "gmflu001_1D": "validation/cmg/flu/gmflu001_1D.sr3",
    "gmflu001": "validation/cmg/flu/gmflu001.sr3",
    "gmflu002_1D": "validation/cmg/flu/gmflu002_1D.sr3",
    "gmflu002": "validation/cmg/flu/gmflu002.sr3",
    "gmflu003_1D": "validation/cmg/flu/gmflu003_1D.sr3",
    "gmflu003": "validation/cmg/flu/gmflu003.sr3",
}


def extract_sr3_values():
    """Extract key values from all SR3 files."""
    parser = SR3Parser()

    results = {}

    for case_id, sr3_path_str in SR3_FILES.items():
        sr3_path = ROOT / sr3_path_str

        print(f"\n{'=' * 60}")
        print(f"Case: {case_id}")
        print(f"SR3 File: {sr3_path}")
        print("=" * 60)

        if not sr3_path.exists():
            print(f"  ERROR: SR3 file not found!")
            results[case_id] = {"error": "File not found"}
            continue

        try:
            data = parser.parse_file(sr3_path)

            if data.get("errors"):
                print(f"  ERROR parsing: {data['errors']}")
                results[case_id] = {"error": data["errors"]}
                continue

            cumulative_oil = data.get("cumulative_oil", 0)
            recovery_factor = data.get("recovery_factor", 0)
            final_pressure = data.get("final_pressure", 0)
            cumulative_gas = data.get("cumulative_gas", 0)
            cumulative_co2_injected = data.get("cumulative_co2_injected", 0)
            cumulative_water = data.get("cumulative_water", 0)

            if recovery_factor > 0:
                ooip_stb = cumulative_oil / recovery_factor
            else:
                ooip_stb = 0

            time_vector = data.get("time_vector", np.array([]))
            pressure_profile = data.get("pressure_profile", np.array([]))
            recovery_profile = data.get("recovery_profile", np.array([]))

            print(f"  Time steps: {len(time_vector)}")
            print(
                f"  Final Time: {time_vector[-1]:.1f} days ({time_vector[-1] / 365.25:.2f} years)"
                if len(time_vector) > 0
                else "  No time data"
            )
            print(f"")
            print(f"  --- CUMULATIVE VALUES ---")
            print(f"  Cumulative Oil:      {cumulative_oil:>15,.0f} STB")
            print(f"  Cumulative Gas:     {cumulative_gas:>15,.0f} SCF")
            print(f"  Cumulative Water:   {cumulative_water:>15,.0f} STB")
            print(f"  Cumulative CO2 Inj: {cumulative_co2_injected:>14,.0f} SCF")
            print(f"")
            print(f"  --- KEY METRICS ---")
            print(f"  Recovery Factor:    {recovery_factor:>14.4f} ({recovery_factor * 100:.2f}%)")
            print(f"  OOIP (derived):     {ooip_stb:>15,.0f} STB")
            print(f"  Final Pressure:    {final_pressure:>15.1f} psia")
            print(f"")
            print(f"  --- PROFILE SHAPES ---")
            if len(pressure_profile) > 0:
                print(
                    f"  Pressure:           min={pressure_profile.min():.1f}, max={pressure_profile.max():.1f}"
                )
            if len(recovery_profile) > 0:
                print(
                    f"  Recovery Profile:   min={recovery_profile.min():.4f}, max={recovery_profile.max():.4f}"
                )

            results[case_id] = {
                "cumulative_oil": cumulative_oil,
                "cumulative_gas": cumulative_gas,
                "cumulative_water": cumulative_water,
                "cumulative_co2_injected": cumulative_co2_injected,
                "recovery_factor": recovery_factor,
                "ooip_stb": ooip_stb,
                "final_pressure": final_pressure,
                "time_steps": len(time_vector),
                "final_time_days": float(time_vector[-1]) if len(time_vector) > 0 else 0,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            results[case_id] = {"error": str(e)}

    return results


def print_summary_table(all_results):
    """Print a summary table of all SR3 values."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: OOIP VALUES FROM SR3 FILES")
    print("=" * 100)
    print(
        f"{'Case':<15} {'OOIP (STB)':>18} {'Cum Oil (STB)':>18} {'RF':>10} {'P_final (psia)':>15}"
    )
    print("-" * 100)

    for case_id, data in all_results.items():
        if "error" in data:
            print(f"{case_id:<15} {'ERROR':>18}")
        else:
            print(
                f"{case_id:<15} {data['ooip_stb']:>18,.0f} {data['cumulative_oil']:>18,.0f} {data['recovery_factor']:>10.4f} {data['final_pressure']:>15.1f}"
            )

    print("=" * 100)


if __name__ == "__main__":
    print("Extracting values from SR3 files...")
    results = extract_sr3_values()
    print_summary_table(results)

    print("\n\nNOTE: These OOIP values are DERIVED from CMG output:")
    print("      OOIP = cumulative_oil / recovery_factor")
    print("\nThese should be used as REFERENCE for validating engine inputs.")
    print("Engine input OOIP should match these values for proper comparison.")
