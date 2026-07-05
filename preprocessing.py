"""
preprocessing.py
-----------------
This file contains all the data cleaning and preprocessing steps
for the Bengaluru House Price Prediction project.

Beginner-friendly note:
We keep every function small and simple so it is easy to follow
what each step is doing to the data.
"""

import pandas as pd
import numpy as np


def load_data(file_path):
    """Load the raw CSV file into a pandas DataFrame."""
    df = pd.read_csv(file_path)
    return df


def drop_unnecessary_columns(df):
    """
    Drop columns that do not help in predicting the price.
    - 'society'      : too many missing values, not useful for prediction.
    - 'availability' : mostly 'Ready To Move', not very useful.
    """
    df = df.drop(columns=['society', 'availability'], errors='ignore')
    return df


def handle_missing_values(df):
    """
    Handle missing values in a simple way:
    - Drop rows where 'location' or 'size' is missing (very few rows).
    - Fill missing 'bath' with the median number of bathrooms.
    - Fill missing 'balcony' with the median number of balconies.
    """
    df = df.dropna(subset=['location', 'size'])

    df['bath'] = df['bath'].fillna(df['bath'].median())
    df['balcony'] = df['balcony'].fillna(df['balcony'].median())

    return df


def convert_size_to_bhk(df):
    """
    Convert the 'size' column (e.g. '2 BHK', '4 Bedroom') into a
    numeric 'bhk' column by extracting the number at the start of the text.
    """
    df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]))
    df = df.drop(columns=['size'])
    return df


def convert_sqft_to_num(x):
    """
    Convert a single total_sqft value into a numeric float.
    Some values are ranges like '2100-2850', so we take the average.
    Some values contain units (e.g. '34.46Sq. Meter') which we cannot
    parse reliably, so we return None for those and drop them later.
    """
    try:
        # Case 1: Simple number, e.g. '1200'
        return float(x)
    except ValueError:
        # Case 2: Range, e.g. '2100 - 2850'
        tokens = str(x).split('-')
        if len(tokens) == 2:
            try:
                return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2
            except ValueError:
                return None
        # Case 3: Any other unusual format -> treat as missing
        return None


def clean_total_sqft(df):
    """Apply the sqft conversion function to the whole column."""
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
    # Drop rows where conversion failed (very small number of rows)
    df = df.dropna(subset=['total_sqft'])
    return df


def clean_location(df):
    """
    Clean the 'location' column:
    - Strip extra spaces.
    - Group rare locations (appearing 10 times or less) into 'other'.
      This reduces the number of categories for One-Hot Encoding and
      avoids the model overfitting to very rare locations.
    """
    df['location'] = df['location'].apply(lambda x: str(x).strip())

    location_counts = df['location'].value_counts()
    rare_locations = location_counts[location_counts <= 10].index

    df['location'] = df['location'].apply(
        lambda x: 'other' if x in rare_locations else x
    )
    return df


def remove_outliers(df):
    """
    Remove obvious outliers to make the model more reliable.
    We keep this simple and rule-based (no advanced statistics):

    1. Remove homes where square feet per bedroom is unusually small
       (less than 300 sqft per BHK is very unlikely to be real).
    2. Remove extreme price_per_sqft outliers using the
       mean and standard deviation within each location.
    """
    # Rule 1: sqft per BHK should be reasonable
    df = df[(df['total_sqft'] / df['bhk']) >= 300]

    # Create a helper column: price per square foot
    df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']

    # Rule 2: Remove rows where price_per_sqft is far from the
    # location's average (keep values within mean +/- 1 std deviation).
    # We use groupby().transform() so each row gets its own location's
    # mean/std lined up next to it, then we filter with simple conditions.
    location_mean = df.groupby('location')['price_per_sqft'].transform('mean')
    location_std = df.groupby('location')['price_per_sqft'].transform('std').fillna(0)

    df = df[
        (df['price_per_sqft'] >= (location_mean - location_std)) &
        (df['price_per_sqft'] <= (location_mean + location_std))
    ]

    # Rule 3: A very simple sanity check on bathrooms
    # (bathrooms should not be more than bhk + 2)
    df = df[df['bath'] < (df['bhk'] + 2)]

    # Drop the helper column, we don't need it for modeling
    df = df.drop(columns=['price_per_sqft'])

    return df


def remove_duplicates(df):
    """Remove exact duplicate rows from the dataset."""
    df = df.drop_duplicates()
    return df


def preprocess_pipeline(file_path):
    """
    Run the full preprocessing pipeline in order and return
    a clean, model-ready DataFrame (before one-hot encoding).
    """
    df = load_data(file_path)
    df = drop_unnecessary_columns(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = convert_size_to_bhk(df)
    df = clean_total_sqft(df)
    df = clean_location(df)
    df = remove_outliers(df)
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Run preprocessing and save the cleaned dataset
    cleaned_df = preprocess_pipeline("Bengaluru_House_Data.csv")
    cleaned_df.to_csv("cleaned_data.csv", index=False)
    print("Preprocessing complete!")
    print("Cleaned data shape:", cleaned_df.shape)
    print(cleaned_df.head())
