import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def generate_watchlist():
    print("Generating Strategic Early-Warning Foresight Watchlist...")

    data_path = 'data/processed/climate_displacement_panel.csv'
    output_dir = 'outputs/reports'
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    features = [
        'disp_risk_hist_3yr',
        'agri_gdp_share_lag1',
        'rural_pop_share_lag1',
        'log_gdp_per_capita_lag1'
    ]
    target_clf = 'high_displacement_alert'

    # Train across full pre-2023 records
    train_df = df[df['year'] < 2023].dropna(subset=[target_clf])
    latest_df = df[df['year'] == 2023].copy()

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(train_df[features]))
    y_train = train_df[target_clf]

    rf = RandomForestClassifier(
        n_estimators=150, max_depth=5, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)

    # Score latest country profiles
    X_latest = scaler.transform(imputer.transform(latest_df[features]))
    latest_df['predicted_alert_probability'] = rf.predict_proba(X_latest)[:, 1]

    # Rank top 20 high-risk countries
    watchlist = latest_df[[
        'economy', 'year', 'predicted_alert_probability', 'disp_per_100k', 'total_population']]
    watchlist = watchlist.sort_values(
        by='predicted_alert_probability', ascending=False).head(20)
    watchlist['predicted_alert_probability'] = (
        watchlist['predicted_alert_probability'] * 100).round(1)

    output_csv = os.path.join(output_dir, 'early_warning_watchlist_2024.csv')
    watchlist.to_csv(output_csv, index=False)

    print(f"\nStrategic Watchlist saved to {output_csv}")
    print("\n--- TOP 10 HIGHEST RISK COUNTRIES (Early-Warning Watchlist) ---")
    print(watchlist[['economy', 'predicted_alert_probability',
          'disp_per_100k']].head(10).to_string(index=False))


if __name__ == '__main__':
    generate_watchlist()
