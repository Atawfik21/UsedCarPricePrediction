import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
# [FIX] Feature Selection
from sklearn.feature_selection import SelectKBest, f_regression 

# --- Algorithms for Person 6 ---
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor

print("--- ⏳ [Person 6] Loading Clean Data... ---")
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

# --- 4 MODELS FOR PERSON 6 ---
# ElasticNet: Mix of Lasso & Ridge
# MLPRegressor: Neural Network (Deep Learning)
models = [
    ("ElasticNet (Default)", ElasticNet(random_state=42)),
    ("ElasticNet (alpha=0.5)", ElasticNet(alpha=0.5, l1_ratio=0.7, random_state=42)),
    ("Neural Network (1 Layer)", MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
    ("Neural Network (2 Layers)", MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
]

print(f"\n{'='*40}")
print(f"--- 🚀 STARTING TRAINING (4 MODELS) ---")
print(f"{'='*40}")

for name, algo in models:
    print(f"\n🔹 [Current Model]: {name}")
    
    pipe = Pipeline(steps=[
        ('prep', preprocessor),
        ('selector', SelectKBest(f_regression, k=500)), # Keep top 500 features for speed
        ('model', algo)
    ])
    
    # Training
    print(f"   ⏳ Phase 1: Fitting/Training model...")
    t0 = time.time()
    pipe.fit(X_train, y_train)
    print(f"   ✅ Fit Complete in {(time.time()-t0):.2f} seconds.")
    
    # Predicting
    print(f"   ⏳ Phase 2: Predicting on Test Data...")
    t1 = time.time()
    y_pred = pipe.predict(X_test)
    
    # Scoring
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"   📊 RESULTS: R2 = {r2:.4f} | MAE = ${mae:.2f}")
    
    # Saving
    filename = f"person6_{name.replace(' ','_').replace('(','').replace(')','').replace('=','')}.joblib"
    print(f"   💾 Saving to {filename}...")
    joblib.dump(pipe, filename)
    print(f"   ✨ {name} DONE.")

print(f"\n{'='*40}")
print("--- ✅ ALL Person 6 Tasks Completed! ---")
print(f"{'='*40}")