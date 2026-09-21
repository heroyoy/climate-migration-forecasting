# Quantitative Climate-Induced Migration Forecasting System
### Early-Warning Foresight & Interactive Monitor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: World Bank & IDMC](https://img.shields.io/badge/Data-World%20Bank%20%7C%20IDMC-green.svg)](#data-architecture)

An end-to-end quantitative forecasting pipeline and diagnostic monitor designed to assess, benchmark, and anticipate climate-induced displacement risks across sovereign states (2011–2024). Built to support strategic policy planning and anticipatory humanitarian action.

---

## Key Empirical Findings

- **Out-of-Time Prospective Evaluation (2020–2023 Test Set):**
  - **Random Forest Early-Warning Classifier:** `ROC-AUC = 0.8437`
  - **Logistic Regression (L2) Baseline:** `ROC-AUC = 0.7913`
  - **Displacement Magnitude Regressor:** `R² = 0.3667` ($MAE = 1.56$)
- **Core Drivers of Vulnerability:**
  1. *Historical 3-Year Displacement Persistence* (54.2% feature importance) confirms climate displacement concentrates in compounding geographic corridors.
  2. *Macroeconomic Buffering Capacity* ($\beta = -0.301$ for Log GDP per Capita) acts as a crucial dampener, enabling local in-situ adaptation.
  3. *Rural Exposure* ($\beta = +0.274$) consistently elevates distress mobility risk.

---

## Project Structure

```text
├── data/
│   ├── raw/                 # Ingested World Bank & IDMC records
│   └── processed/           # Merged country-year panel (2011–2023)
├── outputs/
│   ├── figures/             # Diagnostic plots (Trends, Correlation, ROC curves)
│   ├── metrics/             # Benchmark results and feature importances
│   └── reports/             # 2024–2025 Early-Warning Priority Watchlist
├── app.py                   # Interactive Streamlit monitor & policy sandbox
├── run_pipeline.py          # Master execution orchestrator
├── 01_fetch_worldbank_data.py
├── 02_fetch_idmc_displacement.py
├── 03_merge_and_engineer_features.py
├── 04_exploratory_data_analysis.py
├── 05_train_models.py
├── 06_generate_foresight_report.py
├── requirements.txt
└── README.md