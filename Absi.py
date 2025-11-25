import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
# [FIX] Import Feature Selection to save RAM
from sklearn.feature_selection import SelectKBest, f_regression 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso

print("--- ⏳ [Person 3] Loading Clean Data... ---")
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

print("--- ✂️ Splitting Data... ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("--- 🟢 Data Prepared. Starting Training... ---")

# --- 4 MODELS FOR PERSON 3 (With Feature Selection to fix Memory Error) ---
# We use SelectKBest(k=500) to keep only the top 500 features
models = [
    ("KNN (k=5)", KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),       
    ("KNN (k=15)", KNeighborsRegressor(n_neighbors=15, n_jobs=-1)),     
    ("Lasso (alpha=1.0)", Lasso(alpha=1.0)),                            
    ("Lasso (alpha=0.1)", Lasso(alpha=0.1))                             
]

print(f"\n{'='*40}")
print(f"--- 🚀 STARTING TRAINING (4 MODELS) ---")
print(f"{'='*40}")

for name, algo in models:
    print(f"\n🔹 [Current Model]: {name}")
    
    # [FIX] Added 'selector' step to the pipeline
    pipe = Pipeline(steps=[
        ('prep', preprocessor),
        ('selector', SelectKBest(f_regression, k=500)), # Select top 500 features ONLY
        ('model', algo)
    ])
    
    # Step 2: Training (Fit)
    print(f"   ⏳ Phase 1: Fitting/Training model...")
    t0 = time.time()
    pipe.fit(X_train, y_train)
    print(f"   ✅ Fit Complete in {(time.time()-t0):.2f} seconds.")
    
    # Step 3: Predicting
    print(f"   ⏳ Phase 2: Predicting on Test Data...")
    t1 = time.time()
    y_pred = pipe.predict(X_test)
    print(f"   ✅ Prediction Complete in {(time.time()-t1):.2f} seconds.")
    
    # Step 4: Scoring
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"   📊 RESULTS: R2 = {r2:.4f} | MAE = ${mae:.2f}")
    
    # Step 5: Saving
    filename = f"person3_{name.replace(' ','_').replace('(','').replace(')','').replace('=','')}.joblib"
    print(f"   💾 Saving to {filename}...")
    joblib.dump(pipe, filename)
    print(f"   ✨ {name} DONE.")

print(f"\n{'='*40}")
print("--- ✅ ALL Person 3 Tasks Completed! ---")
print(f"{'='*40}")