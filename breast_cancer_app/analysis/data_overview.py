import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from lifelines import CoxPHFitter


#  cache artefacts
@st.cache_resource
def load_artefacts(model_key, artefact_path):

    model = None
    scaler = None
    feature_cols = None
    numeric_cols = None
    ordinal_cols = None
    nominal_cols = None
    artefacts_exist = os.path.exists(artefact_path)



    if model_key in ["rsf", "cox_ph"]:
        artefacts = joblib.load(artefact_path, mmap_mode='r')
        model = artefacts.get('model')
        scaler = artefacts.get('scaler')
        feature_cols = artefacts.get('feature_cols')
        numeric_cols = artefacts.get('numeric_cols')
        ordinal_cols = artefacts.get("ordinal_cols", [])
        nominal_cols = artefacts.get("nominal_cols", [])

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "ordinal_cols": ordinal_cols,
        "nominal_cols": nominal_cols
    }




def run_data_overview_tab(df, numeric_cols):
    st.header("🔮 Patient Prognosis Interface")


    model_options = {
        "Random Survival Forest": "rsf",
        "Cox Proportional Hazards": "cox_ph",
    }
    selected_model_name = st.selectbox("Choose Survival Model", options=list(model_options.keys()))
    model_key = model_options[selected_model_name]

    artefact_filenames = {
        "rsf": "rsf_artefacts.joblib",
        "cox_ph": "cox_ph_artefacts.joblib",
    }
    artefact_path = os.path.join(
        "breast_cancer_app", "data", "survival_results", model_key, artefact_filenames[model_key]
    )

    loaded_artefacts = load_artefacts(model_key, artefact_path)

    if loaded_artefacts is None:
        if not os.path.exists(artefact_path):
            print(f" Model artefacts not found")


    else:
        model = loaded_artefacts["model"]
        scaler = loaded_artefacts["scaler"]
        feature_cols = loaded_artefacts["feature_cols"]
        numeric_cols = loaded_artefacts["numeric_cols"]
        ordinal_cols = loaded_artefacts["ordinal_cols"]
        nominal_cols = loaded_artefacts["nominal_cols"]

        form_numeric_cols = numeric_cols
        form_categorical_cols = ordinal_cols + nominal_cols

        with st.form(key=f"{model_key}_prediction_form_v2"):
            st.subheader("Input Patient Features")

            raw_user_inputs = {}
            cols_left, cols_right = st.columns(2)

            with cols_left:
                st.markdown("**Numeric Features**")
                for i, col in enumerate(form_numeric_cols):
                    try:
                        default_val = float(df[col].median()) if col in df.columns else 0.0
                    except:
                        default_val = 0.0
                    val = st.number_input(col, value=default_val, key=f"{model_key}_num_{col}")
                    raw_user_inputs[col] = val

            with cols_right:
                st.markdown("**Categorical/Ordinal Features**")
                for i, col in enumerate(form_categorical_cols):
                    if col in df.columns:
                        options = sorted(df[col].dropna().astype(str).unique())
                        try:
                            default_opt = df[col].mode().astype(str).iat[0]
                            default_idx = options.index(default_opt) if default_opt in options else 0
                        except:
                             default_idx = 0
                        sel = st.selectbox(col, options, index=default_idx, key=f"{model_key}_cat_{col}")
                        raw_user_inputs[col] = sel
                    else:
                        print(f"[Internal Warning] Column '{col}' needed for form not found in input dataframe.") # Log internally instead
                        raw_user_inputs[col] = None

            submitted = st.form_submit_button(f"Predict Survival using {selected_model_name}")

        if submitted:

            input_df_raw = pd.DataFrame([raw_user_inputs])

            input_numeric_scaled = scaler.transform(input_df_raw[numeric_cols])
            input_numeric_df = pd.DataFrame(input_numeric_scaled, columns=numeric_cols, index=input_df_raw.index)

            input_ordinal_df = input_df_raw[ordinal_cols]


            input_nominal_df = pd.get_dummies(input_df_raw[nominal_cols], columns=nominal_cols, drop_first=True)

            input_df_combined = pd.concat([input_numeric_df, input_ordinal_df, input_nominal_df], axis=1)


            input_df_processed = input_df_combined.reindex(columns=feature_cols, fill_value=0)



            try:
                if model_key == "rsf":

                        surv_func = model.predict_survival_function(input_df_processed, return_array=False)[0]
                        fig, ax = plt.subplots(figsize=(5,3), dpi=72) # Added dpi
                        ax.step(surv_func.x/12, surv_func.y, where="post")
                        ax.set_xlabel("Years")
                        ax.set_ylabel("Survival Probability")
                        ax.set_title(f"Predicted Survival Curve ({selected_model_name})")
                        ax.set_ylim(0,1)
                        ax.grid(True)
                        st.pyplot(fig, clear_figure=True)


                elif model_key == "cox_ph":
                    surv_func_df = model.predict_survival_function(input_df_processed)
                    surv_probs = surv_func_df.iloc[:, 0]
                    fig, ax = plt.subplots(figsize=(5,3), dpi=72) # Added dpi
                    ax.plot(surv_probs.index/12, surv_probs.values)
                    ax.set_xlabel("Years")
                    ax.set_ylabel("Survival Probability")
                    ax.set_title(f"Predicted Survival Curve ({selected_model_name})")
                    ax.set_ylim(0,1)
                    ax.grid(True)
                    st.pyplot(fig, clear_figure=True)

            except Exception as e:
                print("Prediction failed")
