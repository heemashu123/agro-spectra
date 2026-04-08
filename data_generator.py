"""
=============================================================================
Agro-Spectra | MODULE 1: DATA PIPELINE
File        : data_generator.py
Description : Generates a 365-row synthetic dataset (mock_farm_data.csv)
              representing daily farm telemetry with seasonal climate logic.
              Used as the environment's climate oracle in agro_env.py.
=============================================================================
Usage:
    python data_generator.py
Output:
    mock_farm_data.csv  (saved in the same directory)
=============================================================================
"""

import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Reproducibility seed — set to None for true randomness each run
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Season definitions (day ranges, 1-indexed, inclusive)
# ---------------------------------------------------------------------------
#  Winter  : Days   1 –  59  |  15 – 25 °C  |  Rare 1-5 mm showers
#  Summer  : Days  60 – 149  |  30 – 45 °C  |  Mostly dry, rare showers
#  Monsoon : Days 150 – 250  |  25 – 35 °C  |  High probability 10-50 mm
#  Winter2 : Days 251 – 365  |  15 – 25 °C  |  Rare 1-5 mm showers
# ---------------------------------------------------------------------------

TOTAL_DAYS        = 365
SUMMER_START      = 60
SUMMER_END        = 149
MONSOON_START     = 150
MONSOON_END       = 250

# Rainfall probabilities
P_RAIN_MONSOON    = 0.75   # 75% chance of rain per monsoon day
P_RAIN_DRY        = 0.07   # 7%  chance of a rare shower in dry seasons


def generate_temperature(days: np.ndarray) -> np.ndarray:
    """
    Generate daily temperature (°C) conditioned on season.

    Parameters
    ----------
    days : np.ndarray
        Array of day numbers (1 – 365).

    Returns
    -------
    np.ndarray
        Temperature values (float32) for each day.
    """
    temp = np.zeros(len(days), dtype=np.float32)

    # Boolean season masks
    is_summer  = (days >= SUMMER_START)  & (days <= SUMMER_END)
    is_monsoon = (days >= MONSOON_START) & (days <= MONSOON_END)
    is_winter  = ~is_summer & ~is_monsoon   # everything else

    # Sample temperatures uniformly within each season's range
    temp[is_summer]  = rng.uniform(30.0, 45.0, size=is_summer.sum())
    temp[is_monsoon] = rng.uniform(25.0, 35.0, size=is_monsoon.sum())
    temp[is_winter]  = rng.uniform(15.0, 25.0, size=is_winter.sum())

    return np.round(temp, 2)


def generate_rainfall(days: np.ndarray) -> np.ndarray:
    """
    Generate daily rainfall (mm) conditioned on season.

    Monsoon days: Bernoulli gate (P=0.75) → if rain, draw from Uniform(10, 50).
    Dry days    : Bernoulli gate (P=0.07) → if shower, draw from Uniform(1, 5).

    Parameters
    ----------
    days : np.ndarray
        Array of day numbers (1 – 365).

    Returns
    -------
    np.ndarray
        Rainfall values (float32, ≥ 0) for each day.
    """
    rain = np.zeros(len(days), dtype=np.float32)

    is_monsoon = (days >= MONSOON_START) & (days <= MONSOON_END)
    is_dry     = ~is_monsoon

    # ---- Monsoon rain ----
    monsoon_indices = np.where(is_monsoon)[0]
    rain_event      = rng.random(size=len(monsoon_indices)) < P_RAIN_MONSOON
    rain[monsoon_indices[rain_event]] = rng.uniform(
        10.0, 50.0, size=rain_event.sum()
    )

    # ---- Dry-season rare showers ----
    dry_indices      = np.where(is_dry)[0]
    shower_event     = rng.random(size=len(dry_indices)) < P_RAIN_DRY
    rain[dry_indices[shower_event]] = rng.uniform(
        1.0, 5.0, size=shower_event.sum()
    )

    return np.round(rain, 2)


def generate_ndvi(n: int) -> np.ndarray:
    """
    Generate Sentinel-2 NDVI proxy values.

    NDVI (Normalised Difference Vegetation Index) is drawn uniformly from
    [0.2, 0.8], reflecting varying canopy coverage across the season.

    Parameters
    ----------
    n : int
        Number of values to generate.

    Returns
    -------
    np.ndarray
        NDVI values (float32) in [0.2, 0.8].
    """
    return np.round(rng.uniform(0.2, 0.8, size=n).astype(np.float32), 4)


def generate_dataset() -> pd.DataFrame:
    """
    Assemble the complete 365-day synthetic farm dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Day, Temperature_C, Rainfall_mm,
        Sentinel2_NDVI.
    """
    days = np.arange(1, TOTAL_DAYS + 1)

    df = pd.DataFrame({
        "Day"            : days,
        "Temperature_C"  : generate_temperature(days),
        "Rainfall_mm"    : generate_rainfall(days),
        "Sentinel2_NDVI" : generate_ndvi(TOTAL_DAYS),
    })

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Agro-Spectra | Data Generator")
    print("=" * 60)

    df = generate_dataset()

    # Save to CSV in the same directory as this script
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "mock_farm_data.csv")
    df.to_csv(output_path, index=False)

    print(f"\n[OK] Dataset generated: {output_path}")
    print(f"    Rows : {len(df)}")
    print(f"    Cols : {list(df.columns)}\n")
    print("Sample rows:")
    print(df.head(10).to_string(index=False))
    print("\n--- Season Statistics ---")
    print(df.groupby(
        pd.cut(df["Day"],
               bins=[0, 59, 149, 250, 365],
               labels=["Winter-1", "Summer", "Monsoon", "Winter-2"])
    )[["Temperature_C", "Rainfall_mm"]].describe().round(2))
    print("\n[Done]")
