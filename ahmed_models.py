import pandas as pd
import numpy as np
import time
import joblib # Import joblib for saving models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline # We will use Pipelines

# --- [1] Import Your Algorithms (Standard sklearn/CPU versions) ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor # This will use the CPU
from sklearn.feature_selection import SelectKBest, f_regression # Import Feature Selector

print("--- ⏳ Loading Clean & Sampled Data (cars_cleaned_sampled.csv)... ---")
start_time = time.time()
try:
    df = pd.read_csv('cars_cleaned_sampled.csv')
    print(f"--- 🟢 Load successful! Rows: {len(df)} ---")
except FileNotFoundError:
    print("--- 🔴 ERROR: 'cars_cleaned_sampled.csv' not found. ---")
    print("--- Make sure the clean file is in the same folder as this script ---")
    exit()

# --- Step 1: Pre-flight Clean (Drop 'mpg') ---
if 'mpg' in df.columns:
    df = df.drop('mpg', axis=1)
    print("--- ℹ️ 'mpg' column dropped to simplify preprocessing. ---")

# --- Step 2: Define Features (X) and Target (y) ---
y = df['price']
X = df.drop('price', axis=1)
print(f"--- ℹ️ Target variable 'price' isolated. ---")

# --- Step 3: Define Column Types ---
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
print(f"--- ℹ️ Identified {len(numerical_features)} numerical features and {len(categorical_features)} categorical features. ---")

# --- Step 4: Create the Preprocessing Transformer ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough' 
)

# --- Step 5: Split the Data (Train/Test Split) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"--- 🟢 Data split complete. Training on {len(X_train)} rows... ---")

# ==============================================================================
# [!!!] AHMED'S 4 MODELS (CPU VERSION) [!!!]
# (This will take a long time to run)
# ==============================================================================
all_results = {}
all_pipelines = {} # To store the trained pipelines

# --- [Model 1: Linear Regression (Baseline)] ---
print("\n--- 🚀 [Model 1] Training Linear Regression... ---")
model_1_name = "Linear Regression (Baseline)"
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', LinearRegression())])
pipeline_lr.fit(X_train, y_train)
y_pred_1 = pipeline_lr.predict(X_test)
all_results[model_1_name] = (r2_score(y_test, y_pred_1), mean_absolute_error(y_test, y_pred_1))
all_pipelines[model_1_name] = pipeline_lr # Save pipeline
print(f"--- 🟢 {model_1_name} Trained. ---")


# --- [Model 2: Random Forest (Baseline)] ---
print("\n--- 🚀 [Model 2] Training Random Forest (Baseline, n=100)... (This will take time) ---")
model_2_name = "Random Forest (Baseline, n=100)"
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                               # (n_jobs=-1 uses all available CPU cores)
                               ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
pipeline_rf.fit(X_train, y_train)
y_pred_2 = pipeline_rf.predict(X_test)
all_results[model_2_name] = (r2_score(y_test, y_pred_2), mean_absolute_error(y_test, y_pred_2))
all_pipelines[model_2_name] = pipeline_rf # Save pipeline
print(f"--- 🟢 {model_2_name} Trained. ---")


# --- [Model 3: Random Forest (with Feature Selection)] ---
print("\n--- 🚀 [Model 3] Training Random Forest (on TOP 500 features)... ---")
model_3_name = "RF (Top 500 Features)"
pipeline_rf_kbest = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('selector', SelectKBest(f_regression, k=500)), 
                                    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
pipeline_rf_kbest.fit(X_train, y_train)
y_pred_3 = pipeline_rf_kbest.predict(X_test)
all_results[model_3_name] = (r2_score(y_test, y_pred_3), mean_absolute_error(y_test, y_pred_3))
all_pipelines[model_3_name] = pipeline_rf_kbest # Save pipeline
print(f"--- 🟢 {model_3_name} Trained. ---")


# --- [Model 4: XGBoost (with Feature Selection)] ---
print("\n--- 🚀 [Model 4] Training XGBoost (on TOP 500 features)... ---")
model_4_name = "XGBoost (Top 500 Features)"
pipeline_xgb_kbest = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('selector', SelectKBest(f_regression, k=500)), 
                                     ('model', XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
pipeline_xgb_kbest.fit(X_train, y_train)
y_pred_4 = pipeline_xgb_kbest.predict(X_test)
all_results[model_4_name] = (r2_score(y_test, y_pred_4), mean_absolute_error(y_test, y_pred_4))
all_pipelines[model_4_name] = pipeline_xgb_kbest # Save pipeline
print(f"--- 🟢 {model_4_name} Trained. ---")


# --- Final Results ---
print("\n" + "="*50)
print("--- 📊 FINAL RESULTS (Ahmed's Models) ---")
print("="*50)
for name, (r2, mae) in all_results.items():
    print(f"\n--- 📈 [RESULTS] {name} ---")
    print(f"    R-squared (R2): {r2:.4f}")
    print(f"    Mean Absolute Error (MAE): ${mae:.2f}")

end_time = time.time()
print(f"\n--- ✅ Total script time: {end_time - start_time:.2f} seconds ---")


# --- [NEW] Step 9: Save ALL 4 Models ---
print("\n--- 💾 Saving all 4 models... ---")

# Save Model 1
model_1_pipeline = all_pipelines["Linear Regression (Baseline)"]
model_filename_1 = 'ahmed_model_1_LR.joblib'
joblib.dump(model_1_pipeline, model_filename_1)
print(f"--- 🟢 Saved '{model_filename_1}' ---")

# Save Model 2
model_2_pipeline = all_pipelines["Random Forest (Baseline, n=100)"]
model_filename_2 = 'ahmed_model_2_RF_Baseline.joblib'
joblib.dump(model_2_pipeline, model_filename_2)
print(f"--- 🟢 Saved '{model_filename_2}' ---")

# Save Model 3
model_3_pipeline = all_pipelines["RF (Top 500 Features)"]
model_filename_3 = 'ahmed_model_3_RF_Top500.joblib'
joblib.dump(model_3_pipeline, model_filename_3)
print(f"--- 🟢 Saved '{model_filename_3}' ---")

# Save Model 4
model_4_pipeline = all_pipelines["XGBoost (Top 500 Features)"]
model_filename_4 = 'ahmed_model_4_XGB_Top500.joblib'
joblib.dump(model_4_pipeline, model_filename_4)
print(f"--- 🟢 Saved '{model_filename_4}' ---")