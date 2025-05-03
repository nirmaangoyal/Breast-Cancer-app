import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from .utils import load_and_process_survival_data
from sksurv.metrics import cumulative_dynamic_auc

OUTPUT_DIR   = "breast_cancer_app/data/survival_results/cox_ph"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = load_and_process_survival_data()
print({df.shape})
print(df.columns)

# ----- Define feature columns -----
num_cols = [
    "CS tumor size (2004-2015)",
    "Regional nodes positive (1988+)",
    "Regional nodes examined (1988+)",
    "Age", "Income"
]
ord_cols = ["Grade", "Stage"]
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

# ----- Survival Object -----
y = Surv.from_arrays(event=df['event'].astype(bool), time=df['Survival months'])

# ----- Train/Test Split (on original df) -----
df_train, df_test, y_train, y_test = train_test_split(
    df, y, test_size=0.30, stratify=df['event'], random_state=42
)

# ----- Scaling (Fit on Train, Transform Train & Test) -----
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(df_train[num_cols])
X_test_num_scaled = scaler.transform(df_test[num_cols])

# Convert scaled arrays back to DataFrames with proper index and columns
X_train_num = pd.DataFrame(X_train_num_scaled, columns=num_cols, index=df_train.index)
X_test_num = pd.DataFrame(X_test_num_scaled, columns=num_cols, index=df_test.index)

# ----- One-Hot Encoding (on Train/Test splits) -----
X_train_cat = pd.get_dummies(df_train[NOMINAL], drop_first=True)
X_test_cat = pd.get_dummies(df_test[NOMINAL], drop_first=True)

# Align columns after one-hot encoding - crucial if test set is missing categories present in train
X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='inner', axis=1, fill_value=0)

# ----- Combine Features for Final X_train, X_test -----
X_train = pd.concat([
    X_train_num,
    df_train[ord_cols],
    X_train_cat
], axis=1)

X_test = pd.concat([
    X_test_num,
    df_test[ord_cols],
    X_test_cat
], axis=1)

# ----- Remove zero-variance predictors -----
zero_var_cols = X_train.columns[X_train.nunique() <= 1]
if len(zero_var_cols):
    X_train = X_train.drop(columns=zero_var_cols)
    X_test = X_test.drop(columns=zero_var_cols)
    print(f" Dropped zero-variance columns: {list(zero_var_cols)}")

# ----- Prepare train_df for lifelines fit method -----
train_df_lifelines = X_train.copy()
train_df_lifelines['Survival months'] = y_train['time']
train_df_lifelines['event'] = y_train['event'].astype(int)

# ----- CoxPH Model ----- (Fit using the prepared lifelines DataFrame)
cph = CoxPHFitter(penalizer=1.0)
cph.fit(train_df_lifelines, duration_col='Survival months', event_col='event')

# --- Save Model Artefacts ---
cox_artefacts = {
    'model': cph,
    'scaler': scaler,
    'feature_cols': X_train.columns.tolist(),
    'numeric_cols': num_cols, 
    'ordinal_cols': ord_cols, 
    'nominal_cols': NOMINAL,  
}

artefacts_filename = os.path.join(OUTPUT_DIR, 'cox_ph_artefacts.joblib')
joblib.dump(cox_artefacts, artefacts_filename)


# ----- Feature Importance Plot (Hazard Ratios) -----
plt.figure(figsize=(10, 8))
cph.plot()
plt.title('Hazard Ratios ')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cox_ph_feature_importance.png'), dpi=300)
plt.close()


feature_importance = pd.DataFrame(cph.summary.coef, index=X_train.columns).sort_values(by='coef', ascending=False)
print(feature_importance)

#C-index
pred_risk = -cph.predict_partial_hazard(X_test)
ci = concordance_index(y_test['time'], pred_risk, y_test['event'])
print(f'Concordance index (test set) = {ci:.3f}\n')

km = KaplanMeierFitter()


df_test_copy = df.loc[X_test.index].copy()

# By Stage (using TEST SET data)
plt.figure(figsize=(8,6))
for code,label in {1:'Localized',2:'Regional',3:'Distant'}.items():
    subset_df = df_test_copy[df_test_copy['Stage']==code] # Changed df to df_test_copy
    km.fit(subset_df['Survival months'], subset_df['event'], label=label)
    km.plot(ci_show=False)

plt.xlabel('Months'); plt.ylabel('Survival'); plt.title('KM by Stage (Test Set)') # Updated title
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'km_by_stage.png'), dpi=300)
plt.close()

# By Income Quartile (using TEST SET data)
df_test_copy['Income_q'] = pd.qcut(df_test_copy['Income'],4,labels=['Q1','Q2','Q3','Q4'], duplicates='drop') # Changed df to df_test_copy
plt.figure(figsize=(8,6))
for q in ['Q1','Q2','Q3','Q4']:
    subset_df = df_test_copy[df_test_copy['Income_q']==q] # Changed df to df_test_copy
    km.fit(subset_df['Survival months'], subset_df['event'], label=q)
    km.plot(ci_show=False)

plt.xlabel('Months'); plt.ylabel('Survival'); plt.title('KM by Income (Test Set)') # Updated title
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'km_by_income.png'), dpi=300)
plt.close()

# By Grade (using TEST SET data)
plt.figure(figsize=(8,6))
for g,label in {1:'Grade I',2:'Grade II',3:'Grade III',4:'Grade IV'}.items():
    subset_df = df_test_copy[df_test_copy['Grade']==g] # Changed df to df_test_copy
    km.fit(subset_df['Survival months'], subset_df['event'], label=label)
    km.plot(ci_show=False)
plt.xlabel('Months'); plt.ylabel('Survival'); plt.title('KM by Grade (Test Set)') # Updated title
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'km_by_grade.png'), dpi=300)
plt.close()

# Km by Risk
risk_scores = cph.predict_partial_hazard(X_test)
print("risk_scores", risk_scores)
risk_tertile = pd.qcut(risk_scores, 3, labels=["Low risk", "Medium risk", "High risk"])
test_km_df = pd.DataFrame({
    'time': y_test['time'],
    'event': y_test['event'],
    'risk_group': risk_tertile
})
plt.figure(figsize=(8,6))
for group in ["Low risk", "Medium risk", "High risk"]:
    mask = test_km_df['risk_group'] == group
    km.fit(test_km_df.loc[mask, 'time'], test_km_df.loc[mask, 'event'], label=group)
    km.plot(ci_show=False)
plt.xlabel('Months')
plt.ylabel('Survival Probability')
plt.title('Kaplan-Meier Curves')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'km_by_risk.png'), dpi=300)
plt.close()

# ----- Time-Dependent AUC -----
eval_times = np.arange(12, 12 * 10 + 1, 12) # Months from 1 year to 10 years

cox_risk_scores = cph.predict_partial_hazard(X_test)

# Calculate time-dependent AUC
auc, mean_auc = cumulative_dynamic_auc(
    y_train, 
    y_test, 
    cox_risk_scores, 
    eval_times  
)
print("auc", auc)
print("mean_auc", mean_auc)

# Plotting the time-dependent AUC
plt.figure(figsize=(8, 6))
plt.plot(eval_times / 12, auc, marker="o") # Convert months to years for plotting
plt.axhline(mean_auc, linestyle="--", color="grey", label=f"Mean AUC ({mean_auc:.3f})")
plt.xlabel("Years")
plt.ylabel("Time-dependent AUC")
plt.title("Time-dependent AUC for Cox Model")
plt.legend()
plt.grid(True)
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cox_time_dependent_auc.png'), dpi=300)
plt.close()
