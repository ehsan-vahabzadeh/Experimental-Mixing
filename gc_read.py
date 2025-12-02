import pandas as pd
import os
import numpy as np
def separate_gases_from_csv(path):
    # Read the CSV (no headers assumed)
    df = pd.read_csv(path, sep=r"\t|,", engine="python", skiprows=1)
    
    # Ensure row indices start from 0
    df = df.reset_index(drop=True)
    df = df[["RT [min]", "Area"]].reset_index(drop=True)
    # Separate odd and even indices
    df_H2 = df.iloc[::2].reset_index(drop=True)   # Odd rows (0, 2, 4, ...)
    df_CH4 = df.iloc[1::2].reset_index(drop=True) # Even rows (1, 3, 5, ...)

    # Rename columns for clarity
    df_H2.columns = ["Time_H2", "H2_PeakArea"]
    df_CH4.columns = ["Time_CH4", "CH4_PeakArea"]

    # Combine into one DataFrame (if times align closely)
    df_combined = pd.concat([df_H2, df_CH4], axis=1)

    return df_H2, df_CH4, df_combined


# Example usage
# INPUT_DIR   = r"Y:\Experimental_Mixing\H2-CH4\NOV\Test4"
# path = "H2-CH4-2ml-1bar-Test4.csv"
# path = "CH4-H2-2ml-1bar-Test4.csv"

INPUT_DIR   = r"Y:\Experimental_Mixing\H2-N2\NOV\Test5"
# path = "H2-N2-2ml-1bar-Test5.csv"
path = "N2-H2-2ml-1bar-Test5.csv"
os.chdir(INPUT_DIR)
parts = path.split('-')
Flow_Rate = float(parts[2].replace('ml', ''))
Rock_PV = 7.767227543 # cc 
df_H2, df_CH4, df_combined = separate_gases_from_csv(path)
# Calib_H2 = 2.5389536523517017e-05 * df_H2["H2_PeakArea"] - 0.0007177391512966924 # from H2-CH4 calibration
Calib_H2 = 2.3428170451004368e-05 * df_H2["H2_PeakArea"] + 0.00259851209393841 # from H2-N2 calibration
Time = np.array(df_H2["Time_H2"])
df_H2['inj_PV'] = Flow_Rate * df_H2["Time_H2"] / Rock_PV
df_H2['Calib_H2'] = Calib_H2
new_name = path.replace('.csv', '')
df_H2.to_csv(f"{new_name}-calib.csv", index=False)
 
