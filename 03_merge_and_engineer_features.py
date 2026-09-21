import os
import pandas as pd
import numpy as np


def build_modeling_dataset():
    print("Beginning dataset integration and feature engineering...")

    wb_path = 'data/raw/worldbank_indicators.csv'
    idmc_path = 'data/raw/idmc_displacement_annual.csv'
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, 'climate_displacement_panel.csv')

    wb_df = pd.read_csv(wb_path)
    idmc_df = pd.read_csv(idmc_path)

    # 1. Harmonize keys
    wb_df['economy'] = wb_df['economy'].astype(str).str.strip().str.upper()
    wb_df['year'] = wb_df['year'].astype(int)

    idmc_df['economy'] = idmc_df['economy'].astype(str).str.strip().str.upper()
    idmc_df['year'] = idmc_df['year'].astype(int)

    # 2. Left join: keep full country-year panel
    panel = pd.merge(wb_df, idmc_df, on=['economy', 'year'], how='left')

    # Impute structural zeros for unrecorded displacement events
    panel['new_displacements'] = panel['new_displacements'].fillna(0)

    # 3. Drop microstates or territories with missing populations
    panel = panel.dropna(subset=['total_population'])
    panel = panel[panel['total_population'] > 50000]

    # 4. Construct normalized displacement rate per 100,000 residents
    panel['disp_per_100k'] = (
        panel['new_displacements'] / panel['total_population']) * 100000

    # 5. Sort panel temporally by entity
    panel = panel.sort_values(by=['economy', 'year']).reset_index(drop=True)

    # 6. Feature Engineering: Lags and Rolling Indicators
    # Lag macroeconomic predictors by 1 year (t-1) to predict displacement at (t)
    lag_cols = ['agri_gdp_share', 'rural_pop_share', 'gdp_per_capita']
    for col in lag_cols:
        panel[f'{col}_lag1'] = panel.groupby('economy')[col].shift(1)

    # Historical displacement persistence: 3-year rolling mean of displacement rate
    panel['disp_per_100k_lag1'] = panel.groupby(
        'economy')['disp_per_100k'].shift(1)
    panel['disp_risk_hist_3yr'] = (
        panel.groupby('economy')['disp_per_100k_lag1']
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )

    # 7. Log transformations for skewed variables
    panel['log_gdp_per_capita_lag1'] = np.log1p(panel['gdp_per_capita_lag1'])
    panel['log_disp_per_100k'] = np.log1p(panel['disp_per_100k'])

    # 8. High-Risk Binary Classification Target (Upper decile threshold for policy alerts)
    risk_threshold = panel['disp_per_100k'].quantile(0.90)
    panel['high_displacement_alert'] = (
        panel['disp_per_100k'] >= risk_threshold).astype(int)

    # Drop the first year per country where lag is unavailable (2010)
    final_panel = panel[panel['year'] >= 2011].copy()

    final_panel.to_csv(output_path, index=False)

    print(f"\nIntegration Complete!")
    print(f"Processed file: {output_path}")
    print(
        f"Dimensions: {final_panel.shape[0]} rows, {final_panel.shape[1]} columns")
    print(
        f"90th Percentile Risk Threshold: {risk_threshold:.2f} displacements per 100k citizens")
    print(
        f"Positive Alert Ratio: {final_panel['high_displacement_alert'].mean():.2%}")
    print("\nSample Columns:")
    print(final_panel[['economy', 'year', 'disp_per_100k', 'log_gdp_per_capita_lag1',
          'disp_risk_hist_3yr', 'high_displacement_alert']].head(8))


if __name__ == '__main__':
    build_modeling_dataset()
