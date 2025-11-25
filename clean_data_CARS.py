import pandas as pd
import numpy as np
import time

# --- Step 0: Load Data ---
DATA_FILE = 'cars.csv' # Make sure this is the correct filename from Kaggle
SAMPLE_SIZE = 150000 # We will take a 150k sample to speed up training

print(f"--- ⏳ Loading raw data ({DATA_FILE})... This may take a moment. ---")
start_time = time.time()
try:
    df = pd.read_csv(DATA_FILE)
    print(f"--- 🟢 Load successful! Original rows: {len(df)} ---")
except FileNotFoundError:
    print(f"--- 🔴 ERROR: '{DATA_FILE}' not found. ---")
    print("--- Make sure the file is in the same folder as the script ---")
    exit()

# --- Step 1: Drop Irrelevant Columns ---
print("\n--- 1. Dropping irrelevant columns... ---")
columns_to_drop = [
    'id', 'url', 'model', 'trim', 'body_type', 'vehicle_title', 
    'vin', 'state', 'seller_name', 'seller_rating', 
    'street', 'zip', 'city', 'country' 
]
df_cleaned = df.drop(columns=columns_to_drop, axis=1, errors='ignore')
print(f"--- 🟢 Columns dropped. Remaining columns: {list(df_cleaned.columns)} ---")


# --- Step 2: Clean Outliers (Price and Mileage) ---
print("\n--- 2. Cleaning outliers from 'price' and 'mileage'... ---")
df_cleaned = df_cleaned[(df_cleaned['price'] >= 500) & (df_cleaned['price'] <= 150000)]
df_cleaned = df_cleaned[(df_cleaned['mileage'] >= 1000) & (df_cleaned['mileage'] <= 500000)]
print(f"--- 🟢 Outliers removed. Current logical rows: {len(df_cleaned)} ---")


# --- [FIXED] Step 3: Feature Engineering ---
print("\n--- 3. Creating 'car_age' feature... ---")
current_year = 2024 
# [FIX] The column name is 'year', not 'model_year'
df_cleaned['car_age'] = current_year - df_cleaned['year'] 
# [FIX] Drop the 'year' column
df_cleaned = df_cleaned.drop('year', axis=1, errors='ignore') 
print("--- 🟢 'car_age' created and 'year' dropped. ---")


# --- Step 4: Handle Remaining Missing Values (Imputation) ---
print("\n--- 4. Filling missing data (imputing)... ---")

# 4.1: Fill all categorical NaNs with 'unknown'
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_cleaned[col] = df_cleaned[col].fillna('unknown')

# 4.2: Fill all numerical NaNs with the mean
numerical_cols = df_cleaned.select_dtypes(include=np.number).columns
for col in numerical_cols:
    if col != 'price': # Don't impute the target variable
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())

# 4.3: Drop any remaining rows with NaN (failsafe)
df_cleaned = df_cleaned.dropna()
print(f"--- 🟢 Imputation complete. Total clean rows: {len(df_cleaned)} ---")


# --- Step 5: Take a Random Sample ---
print(f"\n--- 5. Taking a random sample of {SAMPLE_SIZE} rows for faster training... ---")
if len(df_cleaned) > SAMPLE_SIZE:
    df_sampled = df_cleaned.sample(n=SAMPLE_SIZE, random_state=42)
else:
    df_sampled = df_cleaned # Use all if it's already small
    
print(f"--- 🟢 Sampling complete. Final dataset size: {len(df_sampled)} rows ---")


# --- Step 6: Save Clean and Sampled Data ---
print("\n--- 6. Saving clean and sampled file... ---")
output_filename = 'cars_cleaned_sampled.csv'
df_sampled.to_csv(output_filename, index=False)

end_time = time.time()
print(f"\n--- ✅ Task completed successfully in {end_time - start_time:.2f} seconds ---")
print(f"--- 💾 Clean & Sampled data saved to: {output_filename} ---")

print("\n--- ℹ️ Info of clean data (check for NaNs): ---")
df_sampled.info()