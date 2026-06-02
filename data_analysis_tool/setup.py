from setuptools import setup, find_packages

setup(
    name="data_analysis_tool",
    version="0.1.0",
    description="Data Sanitization and Exploration Engine",
    author="Dineth Perera",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.18.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0"
    ],
    python_requires=">=3.10",
)
