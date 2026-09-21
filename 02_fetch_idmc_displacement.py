import os
import glob
import pandas as pd
import numpy as np


def process_idmc_data():
    raw_dir = 'data/raw'
    clean_path = os.path.join(raw_dir, 'idmc_displacement_annual.csv')

    # Look for IDMC file in data/raw/
    candidate_files = glob.glob(f"{raw_dir}/*displacement*.xlsx") + \
        glob.glob(f"{raw_dir}/*displacement*.csv") + \
        glob.glob(f"{raw_dir}/*idmc*.xlsx") + \
        glob.glob(f"{raw_dir}/*idmc*.csv")

    if not candidate_files:
        print("\n[!] No displacement file found in data/raw/.")
        return

    input_file = candidate_files[0]
    print(f"Loading local IDMC file: {input_file}")

    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)

    print(f"Initial raw records loaded: {len(df)}")
    df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]

    # Map columns based on exact schema
    iso_col = next((c for c in ['iso3', 'country_iso3',
                   'iso', 'country_code'] if c in df.columns), None)
    year_col = next(
        (c for c in ['year', 'event_year', 'hazard_year'] if c in df.columns), None)
    disp_col = next((c for c in ['disaster_internal_displacements', 'displacement',
                    'displacements', 'new_displacements'] if c in df.columns), None)
    hazard_col = next((c for c in [
                      'hazard_category', 'hazard_type', 'hazard_sub_type'] if c in df.columns), None)

    print(
        f"Matched columns -> ISO: '{iso_col}', Year: '{year_col}', Displacements: '{disp_col}', Hazard: '{hazard_col}'")

    if not all([iso_col, year_col, disp_col]):
        raise KeyError(
            f"Missing core columns. Detected: iso={iso_col}, year={year_col}, disp={disp_col}")

    # Exclude purely geophysical hazards (e.g., Earthquakes) to isolate weather/climate triggers
    if hazard_col:
        non_climatic = ['earthquake', 'volcanic eruption',
                        'mass movement (dry)', 'geophysical']
        df = df[~df[hazard_col].astype(str).str.lower().isin(non_climatic)]

    # Clean numeric fields
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    df[disp_col] = pd.to_numeric(df[disp_col], errors='coerce').fillna(0)

    # Filter for valid records within 2010-2024
    df = df.dropna(subset=[year_col, iso_col])
    df[year_col] = df[year_col].astype(int)
    df = df[(df[year_col] >= 2010) & (df[year_col] <= 2024)]

    # Aggregate total disaster displacements per country-year
    annual_df = df.groupby([iso_col, year_col])[disp_col].sum().reset_index()
    annual_df.columns = ['economy', 'year', 'new_displacements']

    annual_df.to_csv(clean_path, index=False)
    print(f"\nSuccess! Processed data saved to {clean_path}")
    print(
        f"Dataset shape: {annual_df.shape} (country-year pairs with displacement)")
    print("\nPreview:")
    print(annual_df.head(10))


if __name__ == '__main__':
    process_idmc_data()
