import streamlit as st
import pandas as pd

def set_page_config():
    st.set_page_config(
        page_title="Breast Cancer Prognosis by Nirmaan Goyal",
        page_icon="🔬",
        layout="wide",
    )

def apply_custom_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f5f7f9;
            color: #2c3e50;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            font-weight: 600;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #ffffff;
            border-radius: 4px;
            padding: 10px 10px 0 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4e89ae;
            color: white;
        }
        
        .stSidebar {
            background-color: #e6e9ef;
        }
        .stSidebar .sidebar-content {
            padding: 20px;
        }
        .stSidebar h1, .stSidebar h2, .stSidebar h3 {
            padding-top: 10px;
            color: #2c3e50;
        }
        
        .stProgress > div > div {
            background-color: #4e89ae;
        }
        
        .stMetric {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stMetric label {
            color: #2c3e50;
        }
        
        .dataframe {
            background-color: #ffffff;
            border-radius: 5px;
            border: 1px solid #e1e4e8;
            margin-bottom: 15px;
        }
        .dataframe th {
            background-color: #f0f2f6;
            color: #2c3e50;
            text-align: left;
            padding: 8px;
        }
        .dataframe td {
            padding: 8px;
            border-top: 1px solid #e1e4e8;
        }
        
        .stSelectbox, .stMultiSelect {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 2px;
            margin-bottom: 10px;
        }
        
        .stPlotlyChart {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .stAlert {
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
        }
        
        .description {
            background-color: #ffffff;
            border-left: 4px solid #4e89ae;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 0 5px 5px 0;
            color: #2c3e50;
            font-size: 16px;
            line-height: 1.5;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        [data-testid="stMarkdownContainer"] {
            color: #2c3e50;
            font-size: 16px;
        }
        
        .highlight-stat {
            color: #4e89ae;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

def display_app_header():
    st.title("🔬 Breast Cancer Prognosis: Leveraging Cancer Registry Data for Survival Prediction by Nirmaan Goyal")


