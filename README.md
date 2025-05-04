# Breast Cancer Prognosis Prediction App

## Overview

This project provides a Streamlit-based web application for predicting breast cancer patient prognosis. It utilizes survival analysis models trained on data derived from the SEER Program registry. Users can input patient characteristics and receive a predicted survival curve based on pre-trained models like Cox Proportional Hazards and Random Survival Forest.

## Features

*   **Data Loading & Preprocessing:** Loads and cleans SEER cancer registry data (`export2.csv`).
*   **Survival Model Selection:** Allows users to choose between different pre-trained survival models (CoxPH, RSF).
*   **Patient Feature Input:** Provides an interactive form to input various patient features (numeric, categorical, ordinal).
*   **Survival Prediction:** Predicts patient survival probabilities over time based on the selected model and input features.
*   **Visualization:** Displays the predicted survival curve as a plot.
*   **Interactive Web Interface:** Built with Streamlit for ease of use.

## Data

The primary dataset used is `export2.csv`, located in the `breast_cancer_app/` directory. This dataset was generated and exported using the **SEER\*Stat software**.


## Models

The application utilizes the following survival analysis models:

*   **Cox Proportional Hazards (CoxPH):** Implemented using the `lifelines` library.
*   **Random Survival Forest (RSF):** Implemented using the `scikit-survival` library.

Model training scripts (`cox_ph.py`, `rsf.py`, `deepsurv.py`) are located in `breast_cancer_app/survival_models/`. Trained model artifacts (including the model object, scaler, and feature lists) are saved as `.joblib` files in `breast_cancer_app/data/survival_results/`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the Streamlit application:
    ```bash
    streamlit run breast_cancer_app/app.py
    ```
2.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
3.  Select a survival model and input patient features in the form.
4.  Click the "Predict Survival" button to view the results.



## Key Dependencies

*   Streamlit
*   Pandas
*   NumPy
*   Scikit-learn
*   Scikit-survival
*   Lifelines
*   Matplotlib
*   Joblib
