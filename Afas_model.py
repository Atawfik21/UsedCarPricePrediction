import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# ==============================================================================
# [!!!] AREA 1: PASTE YOUR IMPORTS HERE [!!!]
# (Example: from sklearn.linear_model import Lasso)
# ==============================================================================

from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

# ==============================================================================

print("--- ⏳ Loading Clean Data... ---")
try:
    # Make sure 'cars_cleaned_sampled.csv' is in the same folder!
    df = pd.read_csv('cars_cleaned_sampled.csv')
    print(f"--- 🟢 Load successful! Rows: {len(df)} ---")
except FileNotFoundError:
    print("--- 🔴 ERROR: 'cars_cleaned_sampled.csv' not found. ---")
    exit()

# --- Data Preparation (Standard for everyone) ---
if 'mpg' in df.columns:
    df = df.drop('mpg', axis=1)

y = df['price']
X = df.drop('price', axis=1)

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough' 
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("--- 🟢 Data Prepared. Starting Training... ---")


# ==============================================================================
# [!!!] AREA 2: DEFINE YOUR MODELS HERE [!!!]
# ==============================================================================

# --- MODEL 1 ---
# Change the name and the function below
model_1_name = "XGBoost"
model_1_algo = XGBRegressor(n_estimators=100, random_state=42)              # <--- PASTE YOUR ALGORITHM HERE (e.g., XGBRegressor())

# --- MODEL 2 ---
model_2_name = "Decision Tree"
model_2_algo = DecisionTreeRegressor(random_state=42)              # <--- PASTE YOUR ALGORITHM HERE


# ==============================================================================
# [!!!] DO NOT TOUCH ANYTHING BELOW THIS LINE [!!!]
# ==============================================================================

if model_1_algo is None or model_2_algo is None:
    print("\n--- 🔴 PLEASE EDIT THE CODE TO ADD YOUR MODELS FIRST! ---")
    exit()

models = [(model_1_name, model_1_algo), (model_2_name, model_2_algo)]

for name, algorithm in models:
    print(f"\n--- 🚀 Training {name}... ---")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', algorithm)])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"--- 📈 RESULTS for {name}: ---")
    print(f"    R-squared (R2): {r2:.4f}")
    print(f"    MAE: ${mae:.2f}")
    
    # Save model
    filename = f"{name.replace(' ', '_')}_model.joblib"
    joblib.dump(pipeline, filename)
    print(f"--- 💾 Saved: {filename}")

print("\n--- ✅ DONE! Send these results to the coordinator. ---")