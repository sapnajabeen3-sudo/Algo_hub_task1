"""
train_model.py
---------------
This script:
1. Loads the cleaned dataset.
2. Applies One-Hot Encoding on categorical columns.
3. Splits the data into training and testing sets.
4. Trains a Linear Regression model.
5. Evaluates the model using MAE, MSE, RMSE, and R2 Score.
6. Saves the trained model and the list of training columns
   (needed later by the Streamlit app) using Joblib.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import preprocess_pipeline


def build_features(df):
    """
    Convert categorical columns ('area_type', 'location') into
    numeric columns using One-Hot Encoding.
    'drop_first=True' avoids the dummy variable trap
    (removes one redundant column per category).
    """
    df = pd.get_dummies(df, columns=['area_type', 'location'], drop_first=True)
    return df


def main():
    # Step 1: Load and clean the raw data using our preprocessing pipeline
    print("Loading and cleaning data...")
    df = preprocess_pipeline("Bengaluru_House_Data.csv")

    # Save the cleaned data as well (in case it's not already saved)
    df.to_csv("cleaned_data.csv", index=False)

    # Step 2: One-Hot Encode categorical columns
    print("Encoding categorical features...")
    df_encoded = build_features(df)

    # Step 3: Separate features (X) and target (y)
    X = df_encoded.drop(columns=['price'])
    y = df_encoded['price']

    # Save the column names used for training.
    # The Streamlit app will need this exact column order later.
    model_columns = X.columns.tolist()
    joblib.dump(model_columns, "model_columns.pkl")

    # Step 4: Split data into training and testing sets (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 5: Train the Linear Regression model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 6: Make predictions on the test set
    y_pred = model.predict(X_test)

    # Step 7: Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n----- Model Evaluation -----")
    print(f"MAE  (Mean Absolute Error) : {mae:.2f}")
    print(f"MSE  (Mean Squared Error)  : {mse:.2f}")
    print(f"RMSE (Root Mean Sq. Error) : {rmse:.2f}")
    print(f"R2 Score                   : {r2:.4f}")
    print("-----------------------------\n")

    # Step 8: Save the trained model using Joblib
    joblib.dump(model, "model.pkl")
    print("Model saved as model.pkl")
    print("Model columns saved as model_columns.pkl")


if __name__ == "__main__":
    main()
