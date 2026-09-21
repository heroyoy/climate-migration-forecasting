import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda():
    print("Executing Exploratory Data Analysis & Visualizations...")

    # Paths
    data_path = 'data/processed/climate_displacement_panel.csv'
    fig_dir = 'outputs/figures'
    os.makedirs(fig_dir, exist_ok=True)

    # Load processed panel
    df = pd.read_csv(data_path)

    # Style configuration
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update(
        {'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13})

    # ----------------------------------------------------
    # Plot 1: Global Displacement Trend Over Time (2011-2023)
    # ----------------------------------------------------
    annual_trend = df.groupby(
        'year')['new_displacements'].sum() / 1e6  # in millions

    plt.figure(figsize=(10, 5))
    ax = annual_trend.plot(kind='bar', color='#2b5c8f',
                           edgecolor='black', alpha=0.85)
    plt.title('Figure 1: Total Global Disaster Displacements by Year (2011–2023)',
              pad=15, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Displacements (Millions)')
    plt.xticks(rotation=45)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}M", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/01_global_displacement_trend.png", dpi=300)
    plt.close()
    print("Saved Figure 1: Global annual trend.")

    # ----------------------------------------------------
    # Plot 2: Correlation Heatmap of Risk Predictors
    # ----------------------------------------------------
    feature_cols = [
        'disp_per_100k',
        'disp_risk_hist_3yr',
        'agri_gdp_share_lag1',
        'rural_pop_share_lag1',
        'log_gdp_per_capita_lag1'
    ]
    corr_matrix = df[feature_cols].corr()
    readable_labels = [
        'Displacement Rate',
        'Displacement (3yr Hist)',
        'Agri GDP Share (t-1)',
        'Rural Pop Share (t-1)',
        'Log GDP per Capita (t-1)'
    ]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='vlag',
        xticklabels=readable_labels,
        yticklabels=readable_labels,
        linewidths=0.5,
        vmin=-1, vmax=1
    )
    plt.title('Figure 2: Correlation Matrix of Structural Risk Predictors',
              pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/02_correlation_matrix.png", dpi=300)
    plt.close()
    print("Saved Figure 2: Correlation matrix.")

    # ----------------------------------------------------
    # Plot 3: Vulnerability Distribution - Alert vs Non-Alert
    # ----------------------------------------------------
    clean_subset = df.dropna(
        subset=['agri_gdp_share_lag1', 'high_displacement_alert'])

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        x='high_displacement_alert',
        y='agri_gdp_share_lag1',
        data=clean_subset,
        palette=['#7ea172', '#c7522a']
    )
    plt.xticks([0, 1], ['Baseline / Low Risk (0)',
               'High-Displacement Alert (1)'])
    plt.xlabel('Policy Risk Classification')
    plt.ylabel('Agricultural Share of GDP (% at t-1)')
    plt.title('Figure 3: Agricultural Vulnerability in High Displacement Outliers',
              pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/03_vulnerability_boxplot.png", dpi=300)
    plt.close()
    print("Saved Figure 3: Vulnerability distribution boxplot.")

    print(f"\nEDA Complete! All figures generated in: {fig_dir}")


if __name__ == '__main__':
    run_eda()
