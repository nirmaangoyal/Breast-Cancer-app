import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from lifelines import KaplanMeierFitter
from .utils import load_and_process_survival_data
from sksurv.metrics import cumulative_dynamic_auc

# Model-specific config
OUTPUT_DIR   = "breast_cancer_app/data/survival_results/rsf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and clean data
df = load_and_process_survival_data()
print(df.shape)

num_cols = [
    "CS tumor size (2004-2015)",
    "Regional nodes positive (1988+)",
    "Regional nodes examined (1988+)",
    "Age", "Income"
]
X_num = pd.DataFrame(
    df[num_cols],
    columns=num_cols, index=df.index
)
X_ord = df[["Grade", "Stage"]]
NOMINAL = [
    "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)",
    "Radiation recode",
    "Derived HER2 Recode (2010+)",
    "Chemotherapy recode (yes, no/unk)",
    "ER Status Recode Breast Cancer (1990+)",
    "PR Status Recode Breast Cancer (1990+)",
    "RX Summ--Surg Prim Site (1998+)",
    "Primary Site - labeled",
    "Laterality"
]
X_cat = pd.get_dummies(df[NOMINAL], drop_first=True)
X = pd.concat([X_num, X_ord, X_cat], axis=1)
print(f"X shape: {X.shape}")

# drop zero-variance columns
zero_var_cols = X.columns[X.nunique() <= 1]
if len(zero_var_cols):
    X = X.drop(columns=zero_var_cols)
    print(f" Dropped zero-variance columns: {list(zero_var_cols)}")

# Convert all features to float32 to reduce memory and model size
X = X.astype(np.float32)

y = Surv.from_arrays(event=df["event"].astype(bool), time=df["Survival months"])

# ----- Train/Test Split -----
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.30,
    stratify=df["event"],
    random_state=42
)
print(f" X_tr: {X_tr.shape}, X_te: {X_te.shape}")

scaler = StandardScaler()
scaler.fit(X_tr[num_cols]) 


# ----- MODEL -----
# print("an Fitting Random Survival Forest …")
rsf = RandomSurvivalForest(
    n_estimators=10,        
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    oob_score=True,
    n_jobs=1,               
    random_state=42,
)
rsf.fit(X_tr, y_tr)
print("fit done")
rsf_artefacts = {
    'model': rsf,
    'scaler': scaler,
    'feature_cols': X_tr.columns.tolist(),
    'numeric_cols': num_cols,
    'ordinal_cols': ["Grade", "Stage"], 
    'nominal_cols': NOMINAL 
}
artefacts_filename = os.path.join(OUTPUT_DIR, 'rsf_artefacts.joblib')
joblib.dump(rsf_artefacts, artefacts_filename)
print(f"Trained RSF artefacts saved to {artefacts_filename}")

c_index = rsf.score(X_te, y_te)
print(f"C-index test = {c_index:.3f}")
print(f"OOB concordance = {rsf.oob_score_:.3f}")


risk_scores = rsf.predict(X_te)
risk_tertile = pd.qcut(risk_scores, 3, labels=["Low risk", "Medium risk", "High risk"])
test_km_df = pd.DataFrame({
    'time': y_te['time'],
    'event': y_te['event'].astype(int),
    'risk_group': risk_tertile
})

kmf = KaplanMeierFitter()
plt.figure(figsize=(8,6))
for group in ["Low risk", "Medium risk", "High risk"]:
    mask = test_km_df['risk_group'] == group
    kmf.fit(test_km_df.loc[mask, 'time'], test_km_df.loc[mask, 'event'], label=group)
    kmf.plot(ci_show=False)
plt.xlabel(f"Survival months")
plt.ylabel("Survival Probability")
plt.title("Kaplan-Meier Curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "km_by_risk.png"), dpi=300)
plt.close()


# ----- Time-Dependent AUC -----
eval_times = np.arange(12, 12 * 10 + 1, 12) # Months from 1 year to 10 years

rsf_risk_scores = rsf.predict(X_te)


rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
    y_tr,          
    y_te,          
    rsf_risk_scores, 
    eval_times     
)

# Plotting the time-dependent AUC
print("Plotting the time-dependent AUC", flush=True)
plt.figure(figsize=(8, 6))
plt.plot(eval_times / 12, rsf_auc, marker="o")
plt.axhline(rsf_mean_auc, linestyle="--", color="grey", label=f"Mean AUC ({rsf_mean_auc:.3f})")
plt.xlabel("Years")
plt.ylabel("Time-dependent AUC")
plt.xticks(np.arange(1, 11)) 
plt.title("Time-dependent AUC for Random Survival Forest Model")
plt.legend()
plt.grid(True)
plt.ylim(0.5, 1.0) 
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rsf_time_dependent_auc.png'), dpi=300)
plt.close()

