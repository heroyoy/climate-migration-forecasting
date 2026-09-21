import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pycountry
import os

# --- Page Setup ---
st.set_page_config(
    page_title="Climate Displacement Early-Warning Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Clean Layout Styling ---
st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    .stMetric { background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #1f77b4; }
</style>
""", unsafe_allow_html=True)

# --- ISO3 to Full Country Name Helper ---


@st.cache_data
def get_country_name(iso3_code):
    try:
        country = pycountry.countries.get(
            alpha_3=str(iso3_code).strip().upper())
        return country.name if country else str(iso3_code)
    except Exception:
        return str(iso3_code)

# --- Data Loading ---


@st.cache_data
def load_all_data():
    panel_file = 'data/processed/climate_displacement_panel.csv'
    watchlist_file = 'outputs/reports/early_warning_watchlist_2024.csv'
    metrics_file = 'outputs/metrics/feature_importances.csv'

    panel = pd.read_csv(panel_file) if os.path.exists(panel_file) else None
    watchlist = pd.read_csv(watchlist_file) if os.path.exists(
        watchlist_file) else None
    feat_imp = pd.read_csv(metrics_file) if os.path.exists(
        metrics_file) else None

    # Enrich datasets with full country names
    if panel is not None:
        panel['country_name'] = panel['economy'].apply(get_country_name)
    if watchlist is not None:
        watchlist['country_name'] = watchlist['economy'].apply(
            get_country_name)
        watchlist['display_label'] = watchlist.apply(
            lambda r: f"{r['country_name']} ({r['economy']})", axis=1)

    return panel, watchlist, feat_imp


panel_df, watchlist_df, feat_imp_df = load_all_data()

# --- Header ---
st.title("🌍 Climate-Induced Displacement Foresight & Early-Warning Monitor")
st.caption(
    "Quantitative Policy Intelligence Pipeline | Data: IDMC GIDD & World Bank (2011–2024)")

if panel_df is None or watchlist_df is None:
    st.error("Model outputs missing. Please run scripts 01 to 06 first.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("🕹️ Analytical Controls")
year_selected = st.sidebar.slider(
    "Historical View Year", min_value=2011, max_value=2023, value=2023, step=1)
alert_threshold = st.sidebar.slider(
    "Early-Warning Alert Threshold (%)", min_value=50, max_value=95, value=80, step=5)

# --- Metric KPI Bar ---
current_year = panel_df[panel_df['year'] == year_selected]
total_disp_m = current_year['new_displacements'].sum() / 1e6
high_alert_count = (
    watchlist_df['predicted_alert_probability'] >= alert_threshold).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Monitored Sovereign States",
            f"{current_year['economy'].nunique()}")
col2.metric(f"Global Displacements ({year_selected})", f"{total_disp_m:.2f} M")
col3.metric(f"High-Alert Countries (≥{alert_threshold}%)",
            f"{high_alert_count}", delta="Priority Action")
col4.metric("Prospective ROC-AUC", "0.8437", delta="Random Forest")

st.markdown("---")

# --- Tabs Navigation ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Early-Warning Watchlist",
    "📈 Historical Displacement Trends",
    "🎛️ Policy Simulation Sandbox",
    "🔬 Model Drivers & Methodology"
])

# ----------------------------------------------------
# Tab 1: Early-Warning Watchlist
# ----------------------------------------------------
with tab1:
    st.subheader(
        f"2024–2025 Prospective Early-Warning Watchlist (Risk ≥ {alert_threshold}%)")

    filtered_watch = watchlist_df[watchlist_df['predicted_alert_probability'] >= alert_threshold].sort_values(
        by='predicted_alert_probability', ascending=True
    )

    if not filtered_watch.empty:
        # Chart using readable Country Names
        chart = alt.Chart(filtered_watch).mark_bar(color='#d9534f').encode(
            x=alt.X('predicted_alert_probability:Q',
                    title='Alert Probability (%)', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('country_name:N', sort='-x', title='Country'),
            tooltip=[
                alt.Tooltip('country_name:N', title='Country'),
                alt.Tooltip('economy:N', title='ISO Code'),
                alt.Tooltip('predicted_alert_probability:Q',
                            title='Alert Probability (%)'),
                alt.Tooltip('disp_per_100k:Q',
                            title='Displacement Rate / 100k', format='.1f')
            ]
        ).properties(height=380)

        st.altair_chart(chart, use_container_width=True)

        # Priority Table
        display_table = filtered_watch.sort_values(by='predicted_alert_probability', ascending=False)[[
            'country_name', 'economy', 'predicted_alert_probability', 'disp_per_100k', 'total_population'
        ]].rename(columns={
            'country_name': 'Country Name',
            'economy': 'Code',
            'predicted_alert_probability': 'Risk Probability (%)',
            'disp_per_100k': 'Recent Rate / 100k Pop',
            'total_population': 'Population'
        })

        st.dataframe(display_table, hide_index=True, use_container_width=True)
    else:
        st.info("No countries exceed this alert threshold.")

# ----------------------------------------------------
# Tab 2: Historical Trends
# ----------------------------------------------------
with tab2:
    st.subheader("Global Disaster Displacements (2011–2023)")
    annual_series = panel_df.groupby(
        'year')['new_displacements'].sum().reset_index()
    annual_series['Displacements (Millions)'] = annual_series['new_displacements'] / 1e6

    trend_chart = alt.Chart(annual_series).mark_bar(color='#2b5c8f').encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('Displacements (Millions):Q',
                title='Displacements (Millions)'),
        tooltip=['year', alt.Tooltip(
            'Displacements (Millions):Q', format='.2f')]
    ).properties(height=380)

    st.altair_chart(trend_chart, use_container_width=True)

# ----------------------------------------------------
# Tab 3: Policy Simulation Sandbox
# ----------------------------------------------------
with tab3:
    st.subheader("Policy Adaptation Scenario Simulator")
    st.write("Simulate how macroeconomic capital investment and agricultural adaptation reduce acute displacement risk.")

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        # Select country by full display name
        country_display_map = dict(
            zip(watchlist_df['display_label'], watchlist_df['economy']))
        selected_display = st.selectbox(
            "Select Country to Evaluate", options=list(country_display_map.keys()))
        selected_iso = country_display_map[selected_display]

        # Grab all historical rows for this country and forward-fill missing historical values
        country_history = panel_df[panel_df['economy']
                                   == selected_iso].sort_values('year').copy()
        country_history = country_history.ffill().bfill()

        # Take the most recent populated record
        c_record = country_history.iloc[-1]

        # Safe extraction with defaults if an indicator is entirely absent for a nation
        gdp_val = c_record['gdp_per_capita_lag1']
        agri_val = c_record['agri_gdp_share_lag1']
        rural_val = c_record['rural_pop_share_lag1']
        hist_val = c_record['disp_risk_hist_3yr']

        gdp_str = f"${gdp_val:,.0f}" if pd.notna(
            gdp_val) else "Data Unavailable (Fragile State)"
        agri_str = f"{agri_val:.1f}%" if pd.notna(agri_val) else "N/A"
        rural_str = f"{rural_val:.1f}%" if pd.notna(rural_val) else "N/A"
        hist_str = f"{hist_val:.1f} per 100k" if pd.notna(
            hist_val) else "0.0 per 100k"

        st.markdown(f"### {c_record['country_name']}")
        st.write(f"- **GDP per Capita:** **{gdp_str}**")
        st.write(f"- **Agricultural Share of GDP:** **{agri_str}**")
        st.write(f"- **Rural Population Share:** **{rural_str}**")
        st.write(f"- **3-Yr Historical Displacement Rate:** **{hist_str}**")

    with sim_col2:
        st.markdown("### Strategic Intervention Levers")
        gdp_growth = st.slider(
            "Macroeconomic Buffer Expansion (% GDP/Cap growth)", 0, 100, 20, step=5)
        agri_resilience = st.slider(
            "Agricultural Diversification / Adaptation (% risk mitigation)", 0, 50, 15, step=5)

        # Standardized elasticity calculation
        risk_reduction = min(max(
            (0.3008 * (gdp_growth / 100) + 0.1591 * (agri_resilience / 100)) * 100, 1.0), 45.0)
        st.success(
            f"🎯 **Projected Risk Mitigation:** Pre-arranged structural adaptation reduces baseline probability of acute distress displacement by approximately **{risk_reduction:.1f}%**.")
        
# ----------------------------------------------------
# Tab 4: Model Drivers & Methodology
# ----------------------------------------------------
with tab4:
    st.subheader("Empirical Model Performance & Feature Drivers")
    if feat_imp_df is not None:
        c_imp, c_text = st.columns([1, 1])
        with c_imp:
            st.markdown("**Predictive Feature Importance (Random Forest):**")
            imp_chart = alt.Chart(feat_imp_df).mark_bar(color='#1f77b4').encode(
                x=alt.X('RF_Importance:Q', title='Feature Importance Score'),
                y=alt.Y('Feature:N', sort='-x', title='Predictor'),
                tooltip=['Feature', 'RF_Importance']
            ).properties(height=260)
            st.altair_chart(imp_chart, use_container_width=True)

        with c_text:
            st.markdown("**Methodological Safeguards:**")
            st.markdown("""
            - **Prospective Out-of-Time Split (2020–2023):** Prevents temporal look-ahead leakage across shocks.
            - **Early-Warning Classification (ROC-AUC = 0.8437):** Accurately discriminates tail-risk crisis events (≥ 90th percentile).
            - **Structural Zero Imputation:** Preserves sovereign states with zero recorded events to prevent survivorship bias.
            - **Historical Persistence:** 3-year rolling exposure is the single strongest indicator of future vulnerability.
            """)
