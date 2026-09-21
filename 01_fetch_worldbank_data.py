import wbgapi as wb
import pandas as pd
import os

def fetch_socioeconomic_data():
    print("Connecting to World Bank API...")

    indicators = {
        'NV.AGR.TOTL.ZS': 'agri_gdp_share',
        'SP.RUR.TOTL.ZS': 'rural_pop_share',
        'NY.GDP.PCAP.CD': 'gdp_per_capita',
        'SP.POP.TOTL': 'total_population'
    }

    # Fetch panel data directly: columns='series' places each indicator in its own column
    df = wb.data.DataFrame(
        series=list(indicators.keys()),
        time=range(2010, 2024),
        columns='series',
        numericTimeKeys=True,
        skipAggs=True  # Keeps only sovereign countries, skips global/regional aggregates
    ).reset_index()

    # Rename columns to human-readable names
    df = df.rename(columns={'time': 'year'})
    df = df.rename(columns=indicators)

    # Reorder columns
    ordered_cols = ['economy', 'year'] + list(indicators.values())
    df = df[[col for col in ordered_cols if col in df.columns]]

    # Ensure output directory exists and save
    os.makedirs('data/raw', exist_ok=True)
    output_path = 'data/raw/worldbank_indicators.csv'
    df.to_csv(output_path, index=False)

    print(f"\nSuccess! Data saved to {output_path}")
    print(f"Dataset shape: {df.shape} (rows, columns)")
    print("\nPreview:")
    print(df.head(10))

if __name__ == '__main__':
    fetch_socioeconomic_data()