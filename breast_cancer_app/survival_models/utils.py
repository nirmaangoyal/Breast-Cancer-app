import re
import numpy as np
import pandas as pd

# Shared constants
INPUT_FILE   = "breast_cancer_app/export2.csv"
TIME_COL     = "Survival months"
EVENT_RAW    = "Vital status recode (study cutoff used)"
EXCLUDE = [
    "Year of diagnosis", "Sex", "Marital status at diagnosis",
    "Year of death recode", "Year of follow-up recode"
]
NUMERIC_RAW = [
    "CS tumor size (2004-2015)",
    "Regional nodes positive (1988+)",
    "Regional nodes examined (1988+)"
]
NOMINAL = [
    "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)",
    "Radiation recode",
    "Derived HER2 Recode (2010+)"
]
RARE_THRESH = 0.02
GRADE_MAP = {
    "Well differentiated; Grade I": 1,
    "Moderately differentiated; Grade II": 2,
    "Poorly differentiated; Grade III": 3,
    "Undifferentiated; anaplastic; Grade IV": 4,
}
STAGE_MAP = {"In situ": 0, "Localized": 1, "Regional": 2, "Distant": 3}
AGE_COL    = "Age recode with single ages and 90+"
INCOME_RAW = "Median household income inflation adj to 2022"

def income_to_numeric(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if " - " in s:
        lo, hi = (int(re.sub(r"[^0-9]", "", t)) for t in s.split(" - "))
        return (lo + hi) / 2
    if s.endswith("+"):
        return int(re.sub(r"[^0-9]", "", s[:-1]))
    try:
        return float(re.sub(r"[^0-9]", "", s))
    except ValueError:
        return np.nan


def load_and_process_survival_data():
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(df.shape)

    # Strip whitespace
    str_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Basic conversions
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df["event"] = (df[EVENT_RAW] == "Dead").astype(int)
    print(f"Total events {df['event'].value_counts()}")

    # Drop rows 
    df = df[df[TIME_COL].notna() & df["event"].notna()].copy()

    # Remove excluded columns
    df = df.drop(columns=[c for c in EXCLUDE if c in df.columns], errors='ignore')
    print(f"After exclude: {df.shape}")

    # Feature Engineering
    df["Age"] = pd.to_numeric(df[AGE_COL].str.extract(r"(\d+)")[0], errors="coerce")
    df["Age"].fillna(df["Age"].median(), inplace=True)

    df["Grade"] = df["Grade Recode (thru 2017)"].map(GRADE_MAP)
    df["Grade"].fillna(df["Grade"].median(), inplace=True)
    print(f"Grade distribution: {df['Grade'].value_counts().to_dict()}")

    df["Stage"] = df["Summary stage 2000 (1998-2017)"].map(STAGE_MAP)
    df.loc[df["Stage"].isna(), "Stage"] = df["Stage"].median()
    print(f"Stage distribution: {df['Stage'].value_counts().to_dict()}")

    df["Income"] = df[INCOME_RAW].apply(income_to_numeric)
    df["Income"].fillna(df["Income"].median(), inplace=True)
    print(f"Income distribution: {df['Income'].value_counts().to_dict()}")

    for col in NOMINAL:
        if col in df.columns:
            df[col] = df[col].replace({"Unknown": "Other", "Blank(s)": "Other"})
            freq = df[col].value_counts(normalize=True)
            rare = freq[freq < RARE_THRESH].index
            df[col] = df[col].replace(dict.fromkeys(rare, "Other"))
            print(f"{col} levels → {df[col].value_counts().to_dict()}")



    all_feats = NUMERIC_RAW + ["Age", "Income", "Grade", "Stage"] + NOMINAL
    all_feats = [feat for feat in all_feats if feat in df.columns]
    na_before = df[all_feats].isna().sum().sum()
    print(f"NAs before final drop in predictor columns: {na_before}")
    df.dropna(subset=all_feats, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df 