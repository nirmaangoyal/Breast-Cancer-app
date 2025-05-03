import streamlit as st
import pandas as pd

from breast_cancer_app.survival_models.utils import load_and_process_survival_data
from breast_cancer_app.utils.ui import set_page_config, apply_custom_css, display_app_header
from breast_cancer_app.analysis.data_overview import run_data_overview_tab
from breast_cancer_app.survival_models.utils import NUMERIC_RAW
@st.cache_data
def load_processed_data():
    df = load_and_process_survival_data()
    return df

def main():
    set_page_config()
    apply_custom_css()
    display_app_header()
    
    with st.spinner('Loading and processing dataset...'):
        df = load_processed_data()
    original_numeric_for_display = NUMERIC_RAW + ["Age", "Income"]
    original_numeric_for_display = [c for c in original_numeric_for_display if c in df.columns]
    
    run_data_overview_tab(df, original_numeric_for_display)



if __name__ == '__main__':
    main() 