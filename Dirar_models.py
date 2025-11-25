import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression 
# Algorithms for Person 5
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor

print("--- ⏳ [Person 5] Loading Clean Data... ---")
try:
    df = pd.read_csv('cars_cleaned_sampled.csv')
    print(f"--- 🟢 Load successful! Rows: {len(df)} ---")
except FileNotFoundError:
    print("--- 🔴 ERROR: 'cars_cleaned_sampled.csv' not found. ---")
    exit()

if 'mpg' in df.columns: df = df.drop('mpg', axis=1)
y = df['price']
X = df.drop('price', axis=1)

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ], remainder='passthrough' 
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4 MODELS FOR PERSON 5 ---
models = [
    ("Gradient Boosting (n=100)", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ("Gradient Boosting (n=50)", GradientBoostingRegressor(n_estimators=50, random_state=42)),
    ("Extra Trees (n=100)", ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ("Extra Trees (n=50)", ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1))
]

print(f"\n{'='*40}")
print(f"--- 🚀 STARTING TRAINING (4 MODELS) ---")
print(f"{'='*40}")

for name, algo in models:
    print(f"\n🔹 [Current Model]: {name}")
    
    pipe = Pipeline(steps=[
        ('prep', preprocessor),
        ('selector', SelectKBest(f_regression, k=500)), 
        ('model', algo)
    ])
    
    print(f"   ⏳ Phase 1: Fitting/Training model...")
    t0 = time.time()
    pipe.fit(X_train, y_train)
    print(f"   ✅ Fit Complete in {(time.time()-t0):.2f} seconds.")
    
    print(f"   ⏳ Phase 2: Predicting on Test Data...")
    t1 = time.time()
    y_pred = pipe.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"   📊 RESULTS: R2 = {r2:.4f} | MAE = ${mae:.2f}")
    
    filename = f"person5_{name.replace(' ','_').replace('(','').replace(')','').replace('=','')}.joblib"
    print(f"   💾 Saving to {filename}...")
    joblib.dump(pipe, filename)
    print(f"   ✨ {name} DONE.")

print(f"\n{'='*40}")
print("--- ✅ ALL Person 5 Tasks Completed! ---")
print(f"{'='*40}") 