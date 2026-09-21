# Local Setup, Execution, and Reproduction Guide

This guide contains everything needed to clone, install, run, and launch the **Climate-Induced Displacement Foresight & Early-Warning Monitor**.

---

## 1. Complete Setup & Execution (Windows PowerShell)

```powershell
# Clone repository and enter directory
git clone [https://github.com/](https://github.com/)<your-username>/climate-migration-forecasting.git
cd climate-migration-forecasting

# Initialize virtual environment and allow activation
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

# Upgrade pip and install all required dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run complete pipeline (data fetch, merge, EDA, modeling, foresight report)
python 01_fetch_worldbank_data.py
python 02_fetch_idmc_displacement.py
python 03_merge_and_engineer_features.py
python 04_exploratory_data_analysis.py
python 05_train_models.py
python 06_generate_foresight_report.py

# (Alternative to running individual scripts above):
# python run_pipeline.py

# Launch interactive Streamlit dashboard
streamlit run app.py

# Open in browser:
# http://localhost:8501 or [http://127.0.0.1:8501](http://127.0.0.1:8501)
# Fallback if port 8501 is busy:
# streamlit run app.py --server.port 8502