import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchtuples as tt
import pickle
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from lifelines import KaplanMeierFitter
from .utils import load_and_process_survival_data
from sksurv.metrics import cumulative_dynamic_auc

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
_ = torch.manual_seed(SEED)

# Model-specific config
OUTPUT_DIR   = "breast_cancer_app/data/survival_results/deepsurv"
NET_NODES = [64, 32, 16]
DROPOUT = 0.2
BATCH_SIZE = 128
LEARNING_RATE = 0.005
EPOCHS = 100
PATIENCE = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cpu"

# Load and clean data
df = load_and_process_survival_data()
print(df.shape)

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

y_sksurv = Surv.from_arrays(event=df["event"].astype(bool), time=df["Survival months"])

df_train, df_test, y_train_sksurv, y_test_sksurv = train_test_split(
    df, y_sksurv, test_size=0.30, stratify=df["event"], random_state=SEED
)

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(df_train[num_cols])
X_test_num_scaled = scaler.transform(df_test[num_cols])

X_train_num = pd.DataFrame(X_train_num_scaled, columns=num_cols, index=df_train.index)
X_test_num = pd.DataFrame(X_test_num_scaled, columns=num_cols, index=df_test.index)

X_train_cat = pd.get_dummies(df_train[NOMINAL], drop_first=True)
X_test_cat = pd.get_dummies(df_test[NOMINAL], drop_first=True)

X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='inner', axis=1, fill_value=0)

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

# Remove zero-variance predictors
zero_var_cols = X_train.columns[X_train.nunique() <= 1]
if len(zero_var_cols):
    X_train = X_train.drop(columns=zero_var_cols)
    X_test = X_test.drop(columns=zero_var_cols)
    print(f" Dropped zero-variance columns: {list(zero_var_cols)}")

model_feature_cols = X_train.columns.tolist()


X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

x_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
x_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

y_train_target = (torch.tensor(y_train_sksurv["time"].copy(), dtype=torch.float32),
                  torch.tensor(y_train_sksurv["event"].astype(float).copy(), dtype=torch.float32))
y_test_time_np = y_test_sksurv["time"].astype('float32')
y_test_event_np = y_test_sksurv["event"].astype('float32')


X_train_final_idx, X_val_idx = train_test_split(
    np.arange(len(X_train)), test_size=0.2, random_state=SEED
)

x_train_final_tensor = x_train_tensor[X_train_final_idx]
x_val_tensor = x_train_tensor[X_val_idx]
y_train_final_target = (y_train_target[0][X_train_final_idx], y_train_target[1][X_train_final_idx])
y_val_target = (y_train_target[0][X_val_idx], y_train_target[1][X_val_idx])


# MODEL DEFINITION & TRAINING
in_features = X_train.shape[1]
net = tt.practical.MLPVanilla(
    in_features=in_features,
    num_nodes=NET_NODES,
    out_features=1,
    batch_norm=True,
    dropout=DROPOUT,
    activation=torch.nn.ReLU
).to(device)

optimizer = tt.optim.Adam(lr=LEARNING_RATE)
model = CoxPH(net, optimizer)

print("Training DeepSurv model …")
callbacks = [tt.callbacks.EarlyStopping(patience=PATIENCE)]
log = model.fit(
    x_train_final_tensor,
    y_train_final_target,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=True,
    val_data=(x_val_tensor, y_val_target),
    val_batch_size=BATCH_SIZE
)


x_test_np = X_test.values.astype('float32')
_ = model.compute_baseline_hazards()
surv_df_test = model.predict_surv_df(x_test_np)

# --- Save Model Artefacts ---
if not hasattr(model, 'baseline_hazards_') or model.baseline_hazards_ is None:
    print("Computing baseline hazards before saving...")
    _ = model.compute_baseline_hazards(input=x_val_tensor, target=y_val_target)

ev = EvalSurv(surv_df_test, y_test_time_np, y_test_event_np, censor_surv='km')
c_index_td = ev.concordance_td()
print(f"Concordance Index: {c_index_td:.4f}")

time_grid_ibs = np.linspace(y_test_time_np.min(), y_test_time_np.max(), 100)
ibs = ev.integrated_brier_score(time_grid_ibs)
print(f"Integrated Brier Score: {ibs:.4f}")



partial_hazard = model.predict(x_test_tensor).cpu().detach().numpy().flatten()
risk_tertile = pd.qcut(partial_hazard, 3, labels=["Low risk", "Medium risk", "High risk"], duplicates='drop')

test_km_df = pd.DataFrame({
    'time': y_test_time_np,
    'event': y_test_event_np.astype(bool),
    'risk_group': risk_tertile
})

kmf = KaplanMeierFitter()
plt.figure(figsize=(8,6))
for group in ["Low risk", "Medium risk", "High risk"]:
    subset_df = test_km_df[test_km_df['risk_group'] == group]
    kmf.fit(subset_df['time'], subset_df['event'], label=group)
    kmf.plot(ci_show=False)

plt.xlabel('Months')
plt.ylabel("Survival Probability")
plt.title("Kaplan-Meier Curves by Predicted Risk ")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "km_by_risk.png"), dpi=300)
plt.close()
print(f" KM plot saved to {os.path.join(OUTPUT_DIR, 'km_by_risk.png')}")

kmf_clin = KaplanMeierFitter()


# ----- Time-Dependent AUC -----
eval_times = np.arange(12, 12 * 10 + 1, 12) # Months from 1 year to 10 years

# Calculate time-dependent AUC
print(y_train_sksurv)
print(y_test_sksurv)
deepsurv_auc, deepsurv_mean_auc = cumulative_dynamic_auc(
    y_train_sksurv,
    y_test_sksurv,
    partial_hazard,
    eval_times
)

print("deepsurv_auc", deepsurv_auc)

# Plotting the time-dependent AUC
plt.figure(figsize=(8, 6))
plt.plot(eval_times / 12, deepsurv_auc, marker="o") 
plt.axhline(deepsurv_mean_auc, linestyle="--", color="grey", label=f"Mean AUC ({deepsurv_mean_auc:.3f})")
plt.xlabel("Years")
plt.ylabel("Time-dependent AUC")
plt.title("Time-dependent AUC for DeepSurv Model")
plt.legend()
plt.grid(True)
plt.ylim(0.5, 1.0) 
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'deepsurv_time_dependent_auc.png'), dpi=300)
plt.close()


