import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    mean_absolute_error, root_mean_squared_error, r2_score
)


def train_and_evaluate():
    print("Beginning model training and temporal evaluation...")
    data_path = 'data/processed/climate_displacement_panel.csv'
    metrics_dir = 'outputs/metrics'
    fig_dir = 'outputs/figures'
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    # 1. Feature Definition
    features = [
        'disp_risk_hist_3yr',
        'agri_gdp_share_lag1',
        'rural_pop_share_lag1',
        'log_gdp_per_capita_lag1'
    ]
    target_clf = 'high_displacement_alert'
    target_reg = 'log_disp_per_100k'

    # Filter complete cases on target
    df = df.dropna(subset=[target_clf, target_reg])

    # 2. Chronological Train-Test Split (Train: 2011-2019 | Test: 2020-2023)
    train_mask = df['year'] <= 2019
    test_mask = df['year'] >= 2020

    X_train_raw = df.loc[train_mask, features]
    X_test_raw = df.loc[test_mask, features]

    y_train_clf = df.loc[train_mask, target_clf]
    y_test_clf = df.loc[test_mask, target_clf]

    y_train_reg = df.loc[train_mask, target_reg]
    y_test_reg = df.loc[test_mask, target_reg]

    print(f"Training observations (2011-2019): {len(X_train_raw)}")
    print(f"Testing observations  (2020-2023): {len(X_test_raw)}")

    # 3. Preprocessing: Median Imputation + Standardization
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw))
    X_test = scaler.transform(imputer.transform(X_test_raw))

    # ====================================================
    # TASK 1: Early-Warning Classification (Alert Risk)
    # ====================================================
    print("\n--- [Task 1: Early-Warning Alert Classification] ---")

    # 1A. Logistic Regression Baseline
    lr = LogisticRegression(class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train_clf)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test_clf, lr_probs)

    # 1B. Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    rf_clf.fit(X_train, y_train_clf)
    rf_probs = rf_clf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test_clf, rf_probs)

    print(f"Logistic Regression ROC-AUC : {lr_auc:.4f}")
    print(f"Random Forest Classifier AUC: {rf_auc:.4f}")

    # Plot ROC Curves
    plt.figure(figsize=(7, 6))
    fpr_lr, tpr_lr, _ = roc_curve(y_test_clf, lr_probs)
    fpr_rf, tpr_rf, _ = roc_curve(y_test_clf, rf_probs)

    plt.plot(fpr_lr, tpr_lr,
             label=f'Logistic Regression (AUC = {lr_auc:.2f})', color='#3b6978', lw=2)
    plt.plot(fpr_rf, tpr_rf,
             label=f'Random Forest (AUC = {rf_auc:.2f})', color='#e05a47', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance')
    plt.title('Figure 4: Prospective ROC Performance (2020–2023 Out-of-Time Test)',
              pad=12, fontweight='bold')
    plt.xlabel('False Positive Rate (False Alarm)')
    plt.ylabel('True Positive Rate (Detection Recall)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/04_roc_curves.png", dpi=300)
    plt.close()

    # ====================================================
    # TASK 2: Continuous Displacement Intensity Estimation
    # ====================================================
    print("\n--- [Task 2: Continuous Displacement Regression] ---")

    # 2A. Ridge Regression Baseline
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train_reg)
    ridge_preds = ridge.predict(X_test)
    ridge_mae = mean_absolute_error(y_test_reg, ridge_preds)
    ridge_rmse = root_mean_squared_error(y_test_reg, ridge_preds)
    ridge_r2 = r2_score(y_test_reg, ridge_preds)

    # 2B. Random Forest Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=150, max_depth=5, random_state=42)
    rf_reg.fit(X_train, y_train_reg)
    rf_preds = rf_reg.predict(X_test)
    rf_mae = mean_absolute_error(y_test_reg, rf_preds)
    rf_rmse = root_mean_squared_error(y_test_reg, rf_preds)
    rf_r2 = r2_score(y_test_reg, rf_preds)

    print(
        f"Ridge Regression     -> MAE: {ridge_mae:.4f} | RMSE: {ridge_rmse:.4f} | R²: {ridge_r2:.4f}")
    print(
        f"Random Forest Reg    -> MAE: {rf_mae:.4f} | RMSE: {rf_rmse:.4f} | R²: {rf_r2:.4f}")

    # ====================================================
    # Feature Importance Export (Policy Insights)
    # ====================================================
    importance_df = pd.DataFrame({
        'Feature': [
            'Historical 3-Yr Rate',
            'Agri GDP Share (t-1)',
            'Rural Pop Share (t-1)',
            'Log GDP per Cap (t-1)'
        ],
        'Logistic_Coef': lr.coef_[0],
        'RF_Importance': rf_clf.feature_importances_
    }).sort_values(by='RF_Importance', ascending=False)

    importance_df.to_csv(f"{metrics_dir}/feature_importances.csv", index=False)
    print("\nModel Risk Drivers & Feature Importances:")
    print(importance_df)

    # Save summary metrics table
    results_summary = pd.DataFrame([
        {'Model': 'Logistic Regression', 'Task': 'Classification',
            'Primary Metric (AUC)': lr_auc, 'Secondary Metric': 'N/A'},
        {'Model': 'Random Forest Classifier', 'Task': 'Classification',
            'Primary Metric (AUC)': rf_auc, 'Secondary Metric': 'N/A'},
        {'Model': 'Ridge Regression', 'Task': 'Regression',
            'Primary Metric (R²)': ridge_r2, 'Secondary Metric (RMSE)': ridge_rmse},
        {'Model': 'Random Forest Regressor', 'Task': 'Regression',
            'Primary Metric (R²)': rf_r2, 'Secondary Metric (RMSE)': rf_rmse}
    ])
    results_summary.to_csv(f"{metrics_dir}/benchmark_results.csv", index=False)
    print(f"\nEvaluation metrics successfully stored in {metrics_dir}/")


if __name__ == '__main__':
    train_and_evaluate()
