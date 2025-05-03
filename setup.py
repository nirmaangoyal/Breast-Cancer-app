from setuptools import setup, find_packages

setup(
    name="breast_cancer_app",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn==1.3.2",
        "joblib",
        "lifelines",
        "plotly",
        "scikit-survival==0.22.1",
    ],
) 
