"""this script generates synthetic housing data for regression tasks of the data collection phase """

import pandas as pd
import numpy as np
import random
from datetime import datetime

def generate_synthetic_housing_data(n_rows=1500, missing_rate=0.15, duplicate_rate=0.08):
    
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    print(f"Generating {n_rows} rows of synthetic housing data...")
    
    # Generate base features
    data = {}
    
    # 1. Square footage (500 to 5000 sq ft)
    data['sqft_living'] = np.random.normal(2000, 800, n_rows).astype(int)
    data['sqft_living'] = np.clip(data['sqft_living'], 500, 5000)
    
    # 2. Number of bedrooms (1 to 6)
    data['bedrooms'] = np.random.choice([1, 2, 3, 4, 5, 6], n_rows, 
                                       p=[0.05, 0.15, 0.35, 0.30, 0.12, 0.03])
    
    # 3. Number of bathrooms (1 to 4.5, in 0.5 increments)
    bath_options = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    data['bathrooms'] = np.random.choice(bath_options, n_rows,
                                        p=[0.08, 0.12, 0.25, 0.20, 0.15, 0.10, 0.07, 0.03])
    
    # 4. Age of house (0 to 100 years)
    data['age'] = np.random.exponential(15, n_rows).astype(int)
    data['age'] = np.clip(data['age'], 0, 100)
    
    # 5. Lot size (1000 to 50000 sq ft)
    data['sqft_lot'] = np.random.lognormal(9, 0.8, n_rows).astype(int)
    data['sqft_lot'] = np.clip(data['sqft_lot'], 1000, 50000)
    
    # 6. Number of floors (1 to 3)
    data['floors'] = np.random.choice([1, 1.5, 2, 2.5, 3], n_rows,
                                     p=[0.40, 0.15, 0.30, 0.10, 0.05])
    
    # 7. Condition (1-5 scale, 1=poor, 5=excellent)
    data['condition'] = np.random.choice([1, 2, 3, 4, 5], n_rows,
                                        p=[0.05, 0.15, 0.50, 0.25, 0.05])
    
    # 8. Grade (quality of construction, 1-13 scale)
    data['grade'] = np.random.choice(range(1, 14), n_rows,
                                    p=[0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 
                                       0.15, 0.10, 0.05, 0.03, 0.01, 0.01])
    
    # 9. Price (target variable) - based on other features with some noise
    # Create a realistic relationship between features and price
    base_price = (
        data['sqft_living'] * 150 +  # $150 per sq ft
        data['bedrooms'] * 15000 +   # $15k per bedroom
        data['bathrooms'] * 20000 +  # $20k per bathroom
        (101 - data['age']) * 2000 + # Newer houses worth more
        data['sqft_lot'] * 2 +       # $2 per sq ft of lot
        data['floors'] * 25000 +     # $25k per floor
        data['condition'] * 40000 +  # $40k per condition point
        data['grade'] * 30000        # $30k per grade point
    )
    
    # Add some random noise to make it more realistic
    noise = np.random.normal(0, 50000, n_rows)
    data['price'] = base_price + noise
    data['price'] = np.clip(data['price'], 50000, 2000000).astype(int)  # Reasonable price range
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print(f"Generated clean dataset with shape: {df.shape}")
    
    # Introduce duplicate rows
    n_duplicates = int(n_rows * duplicate_rate)
    if n_duplicates > 0:
        duplicate_indices = np.random.choice(df.index, n_duplicates, replace=True)
        duplicate_rows = df.loc[duplicate_indices].copy()
        df = pd.concat([df, duplicate_rows], ignore_index=True)
        print(f"Added {n_duplicates} duplicate rows")
    
    # Introduce missing values randomly
    total_cells = df.shape[0] * df.shape[1]
    n_missing = int(total_cells * missing_rate)
    
    for _ in range(n_missing):
        row_idx = np.random.randint(0, df.shape[0])
        col_idx = np.random.randint(0, df.shape[1])
        df.iloc[row_idx, col_idx] = np.nan
    
    print(f"Introduced {n_missing} missing values ({missing_rate*100:.1f}% of total cells)")
    print(f"Final dataset shape: {df.shape}")
    print(f"Missing values per column:")
    print(df.isnull().sum())
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    
    return df

def save_data_to_csv(df, filename="housing_data.csv"):
    """Save the generated data to CSV file in the data directory."""
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    filepath = os.path.join("data", filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to: {filepath}")
    
    return filepath

def load_data(filename="housing_data.csv"):
    """Load the housing data from CSV file."""
    import os
    
    filepath = os.path.join("data", filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"Data loaded from: {filepath}")
        print(f"Dataset shape: {df.shape}")
        return df
    else:
        print(f"File not found: {filepath}")
        return None

def get_data_summary(df):
    """Get summary statistics and data quality info."""
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    print(f"Dataset shape: {df.shape}")
    print(f"Total missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    print("\nColumn Information:")
    print(df.info())
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nMissing Values by Column:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0])

if __name__ == "__main__":
    # Generate the synthetic data
    housing_df = generate_synthetic_housing_data(
        n_rows=1500,
        missing_rate=0.15,  # 15% missing values
        duplicate_rate=0.08  # 8% duplicate rows
    )
    
    # Save to CSV
    save_data_to_csv(housing_df)
    
    # Display summary
    get_data_summary(housing_df)
    
    print("\n" + "="*50)
    print("Sample of the generated data:")
    print("="*50)
    print(housing_df.head(10))