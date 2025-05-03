from setuptools import setup, find_packages

setup(
    name="breast_cancer_app",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "lifelines",
    ],
) 