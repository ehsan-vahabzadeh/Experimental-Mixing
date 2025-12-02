import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

# ======== CONFIG ========
INPUT_DIR   = r"Y:\Experimental_Mixing\H2-N2\NOV\Calibration"
os.chdir(INPUT_DIR)
XLSX_PATH ="H2-N2-Calib.xlsx"   # <-- your file
SKIP_ROWS = 2                        # skip header rows so row 3 is the first data row (0-based)
ENGINE = "openpyxl"                  # or "calamine" if openpyxl gives style errors

# Fixed Excel columns (letters) for this sheet:
# Sample names (A), CH4 Peak Area replicates are at F, I, L; H2 Peak Area replicates at O, R, U.
# (Rationale: each replicate block is 3 columns: RT, Peak Area, Conc. Peak Area is the 2nd.)
COL_SAMPLE = "A"
CH4_PEAK_COLS = ["E", "H", "K"]     # CH4-1, CH4-2, CH4-3 (Peak Area columns)
H2_PEAK_COLS  = ["N", "Q", "T"]     # H2-1,  H2-2,  H2-3  (Peak Area columns)

# If you later add another gas (e.g., CO2), add its Peak Area letters here and extend logic below.
# ========================

# Regex to parse names like:
# Calib-H2-0.2ml-CH4-3.8ml-2025-11-05 11:06:45+00:00
# (tolerate 'ml' or 'l' typos)
name_pat = re.compile(
    r"Calib-(?P<g1>[A-Za-z0-9]+)-(?P<v1>[\d\.]+)m?l-(?P<g2>[A-Za-z0-9]+)-(?P<v2>[\d\.]+)m?l-",
    flags=re.IGNORECASE
)

def parse_name(sample: str):
    """Return (gasA, volA_ml, gasB, volB_ml) parsed from sample name; else None."""
    if not isinstance(sample, str):
        return None
    s = " ".join(sample.split())
    m = name_pat.search(s)
    if not m:
        return None
    g1 = m.group("g1").upper()
    g2 = m.group("g2").upper()
    v1 = float(m.group("v1"))
    v2 = float(m.group("v2"))
    return g1, v1, g2, v2

def main():
    # Build usecols string like "A,F,I,L,O,R,U"
    usecols = [COL_SAMPLE] + CH4_PEAK_COLS + H2_PEAK_COLS
    usecols_str = ",".join(usecols)

    # Read as raw values; no headers from file
    raw = pd.read_excel(
        XLSX_PATH,
        header=None,
        usecols=usecols_str,
        skiprows=SKIP_ROWS,
        engine=ENGINE
    )

    # Name the columns
    col_names = (
        ["Sample"]
        + [f"CH4_rep{i+1}" for i in range(len(CH4_PEAK_COLS))]
        + [f"H2_rep{i+1}"  for i in range(len(H2_PEAK_COLS))]
    )
    raw.columns = col_names
    # Drop fully empty rows (in case trailing blanks)
    raw = raw.dropna(subset=["Sample"], how="all")
    # Average replicates (ignore NaNs)
    raw["CH4_area"] = raw[[c for c in raw.columns if c.startswith("CH4_rep")]] \
    .fillna(0).mean(axis=1)

    raw["H2_area"] = raw[[c for c in raw.columns if c.startswith("H2_rep")]] \
        .fillna(0).mean(axis=1)

    # Parse sample names → volumes and target fractions
    parsed = raw["Sample"].apply(parse_name)
    parsed_df = parsed.apply(
        lambda x: pd.Series(x, index=["gas_A","vol_A_ml","gas_B","vol_B_ml"]) if x else pd.Series([np.nan]*4, index=["gas_A","vol_A_ml","gas_B","vol_B_ml"])
    )
    df = pd.concat([raw, parsed_df], axis=1).dropna(subset=["gas_A","gas_B"])
    df = df[~((df["vol_A_ml"] + df["vol_B_ml"] < 2) | (df["vol_A_ml"] + df["vol_B_ml"] > 2))]

    df["tot_ml"] = df["vol_A_ml"] + df["vol_B_ml"]
    df["frac_A_target"] = df["vol_A_ml"] / df["tot_ml"]
    df["frac_B_target"] = df["vol_B_ml"] / df["tot_ml"]

    # Map the measured areas to gas_A / gas_B, regardless of which is H2 or CH4
    def pick_areas(row):
        a = row["gas_A"]
        b = row["gas_B"]
        # Known measured gases in this sheet are H2 and CH4:
        g2area = {"H2": row["H2_area"], "CH4": row["CH4_area"]}
        area_A = g2area.get(a, np.nan)
        area_B = g2area.get(b, np.nan)
        return pd.Series({"area_A": area_A, "area_B": area_B})
    df = pd.concat([df, df.apply(pick_areas, axis=1)], axis=1)

    # Area ratio referenced to gas_A
    df["area_ratio"] = df["area_A"] / (df["area_A"] + df["area_B"])

    # Tidy selection
    tidy = df[[
        "Sample",
        "gas_A","vol_A_ml","gas_B","vol_B_ml",
        "frac_A_target","frac_B_target",
        "CH4_area","H2_area",
        "area_A","area_B","area_ratio"
    ]].reset_index(drop=True)
    # Save clean file
    tidy.to_csv("gc_calib_clean.csv", index=False)
    print("Saved → gc_calib_clean.csv")
    print(tidy.head(5))
    
    
    
    X = df["area_A"].values.reshape(-1, 1)
    Y = df ["frac_A_target"].values
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, Y)
    a = model.coef_[0]
    b = model.intercept_
    x_range = np.linspace(0, np.max(X), 10000).reshape(-1, 1)
    y_pred = model.predict(x_range)

    plt.figure(figsize=(7,5))
    plt.scatter(X, Y, color="blue", label="Experimental data")
    plt.plot(x_range, y_pred, color="red", lw=2, label=f"Fit: y = {a:.7f}x + {b:.7f}")
    plt.xlabel("Peak Area")
    plt.ylabel("Fraction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
