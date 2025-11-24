# plot_manual_pairs.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_manual_pairs(df) -> pd.DataFrame:

    # Ensure 6 columns exist
    print (df.shape[1])
    while df.shape[1] < 6:
        df[df.shape[1]] = np.nan
    df = df.iloc[:, :12]
    df.columns = ["x1","y1","x2","y2","x3","y3","x4","y4","x5","y5","x6","y6"]

    # Drop rows that are entirely empty
    df = df.dropna(how="all")

    # Optional: sort each pair by x (ignoring NaNs), so lines are monotone in x
    for (xcol, ycol) in [("x1","y1"),("x2","y2"),("x3","y3")]:
        mask = df[[xcol, ycol]].notna().all(axis=1)
        if mask.any():
            sub = df.loc[mask, [xcol, ycol]].sort_values(xcol)
            df.loc[mask, [xcol, ycol]] = sub.values

    return df
def plot_pairs_all(df: pd.DataFrame, out_png: str | None = None, title="Manual pairs"):
    plt.figure(figsize=(7,5))

    for label, (xcol, ycol) in {
        "Loading 2": ("x2","y2"),
        "Loading 3": ("x3","y3"),
        "Unloading 1": ("x4","y4"),
        "Unloading 2": ("x5","y5"),
        "Unloading 3": ("x6","y6"),
    }.items():
        mask = df[[xcol, ycol]].notna().all(axis=1)
        if mask.any():
            if "Loading" in label:
                plt.plot(df.loc[mask, xcol], df.loc[mask, ycol], label=label)
            else:
                plt.plot(df.loc[mask, xcol], 1 - df.loc[mask, ycol], linestyle = "--", label=label)

    plt.xlabel("Injected PV [-]")
    plt.ylabel("Invading Concentration [-]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"Saved plot → {out_png}")
    else:
        plt.show()
        
def plot_pairs_unloading(df: pd.DataFrame, out_png: str | None = None, title="Manual pairs"):
    plt.figure(figsize=(7,5))

    for label, (xcol, ycol) in {
        "Unloading 1": ("x4","y4"),
        "Unloading 2": ("x5","y5"),
        "Unloading 3": ("x6","y6"),
    }.items():
        mask = df[[xcol, ycol]].notna().all(axis=1)
        if mask.any():
            plt.plot(df.loc[mask, xcol], 1 - df.loc[mask, ycol], label=label)

    plt.xlabel("Injected PV [-]")
    plt.ylabel("CH4 Concentration [-]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"Saved plot → {out_png}")
    else:
        plt.show()
        
        
def plot_pairs(df: pd.DataFrame, out_png: str | None = None, title="Manual pairs"):
    plt.figure(figsize=(7,5))

    for label, (xcol, ycol) in {
        "Loading 1": ("x1","y1"),
        "Loading 2": ("x2","y2"),
        "Loading 3": ("x3","y3"),
    }.items():
        mask = df[[xcol, ycol]].notna().all(axis=1)
        if mask.any():
            plt.plot(df.loc[mask, xcol], df.loc[mask, ycol], label=label)

    plt.xlabel("Injected PV [-]")
    plt.ylabel("H2 Concentration [-]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"Saved plot → {out_png}")
    else:
        plt.show()

if __name__ == "__main__":
    # change this to your file path
    path = "manual_matrix.txt"
    INPUT_DIR   = r"Y:\Experimental_Mixing\H2-CH4"
    import os
    os.chdir(INPUT_DIR)
    path = "Final.csv"
    df = pd.read_csv(path, sep=r"\t|,", engine="python", skiprows=1,na_values=["", " ", "NA", "NaN", None])
    df = load_manual_pairs(df)
    plot_pairs(df, out_png="manual_pairs.png", title="Loading/Unloading pairs")
    plot_pairs_unloading(df, out_png="manual_pairs_unloading.png", title="Loading/Unloading pairs")
    plot_pairs_all(df, out_png="manual_pairs_all.png", title="Loading/Unloading pairs")
