import pandas as pd
import numpy as np
import joblib

print("--- 🤖 Loading Ahmed's 4 Saved Models... ---")
try:
    # Load all 4 models
    model_1_lr = joblib.load('ahmed_model_1_LR.joblib')
    model_2_rf_base = joblib.load('ahmed_model_2_RF_Baseline.joblib')
    model_3_rf_top500 = joblib.load('ahmed_model_3_RF_Top500.joblib')
    model_4_xgb_top500 = joblib.load('ahmed_model_4_XGB_Top500.joblib')
    print("--- 🟢 All 4 models loaded successfully! ---")
except FileNotFoundError:
    print("--- 🔴 ERROR: Model files not found. ---")
    print("--- Make sure you ran 'ahmed_models.py' first to save the models. ---")
    exit()

# ==============================================================
# --- [!!!] TEST YOUR OWN CAR HERE [!!!] ---
# (Change the values below to predict a price)
# ==============================================================

my_car = {
    'manufacturer': 'kia',
    'mileage': 320000.0,
    'engine': 'V6',
    'transmission': 'Automatic',
    'drivetrain': '4WD',
    'fuel_type': 'Gasoline',
    'exterior_color': 'Black',
    'interior_color': 'Black',
    'accidents_or_damage': 0.0,
    'one_owner': 1.0,
    'personal_use_only': 1.0,
    'driver_rating': 4.5,
    'driver_reviews_num': 1500.0,
    'price_drop': 0.0,
    'car_age': 5.0 # (e.g., 2024 - 2019)
}
# ==============================================================

# Convert the dictionary to a DataFrame (1 row)
my_car_df = pd.DataFrame([my_car])

print(f"\n--- 🚗 Predicting price for a '{my_car['manufacturer']}' with {my_car['mileage']} miles... ---")

# --- Run Prediction on ALL 4 models ---
pred_1 = model_1_lr.predict(my_car_df)[0]
pred_2 = model_2_rf_base.predict(my_car_df)[0]
pred_3 = model_3_rf_top500.predict(my_car_df)[0]
pred_4 = model_4_xgb_top500.predict(my_car_df)[0]


print("\n" + "="*50)
print("--- 💰 PREDICTION RESULTS (COMPARISON) ---")
print("="*50)
print(f"    Model 1 (Linear Regression):        ${pred_1:,.2f}")
print(f"    Model 2 (Baseline RF - all features): ${pred_2:,.2f}")
print(f"    Model 3 (RF - Top 500 features):      ${pred_3:,.2f}")
print(f"    Model 4 (XGBoost - Top 500 features): ${pred_4:,.2f}")