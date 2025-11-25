import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
import joblib

CSV_PATH = "cars_cleaned_sampled.csv"

print("\n--- ⏳ Loading Clean Data... ---")
df = pd.read_csv(CSV_PATH)
print(f"--- 🟢 Load successful! Rows: {len(df)} ---")

if "mpg" in df.columns:
    df = df.drop("mpg", axis=1)

y = df["price"]
X = df.drop("price", axis=1)

# === No OneHot. Safe ===
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

for c in X.columns:
    X[c] = pd.to_numeric(X[c], downcast="float")
y = pd.to_numeric(y, downcast="float")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("--- 🟢 Data Prepared. Starting Training... ---")

# === ULTRA FAST GB (5k rows only) ===
gb_n = min(5000, len(X_train))
X_train_gb = X_train.sample(gb_n, random_state=42)
y_train_gb = y_train.loc[X_train_gb.index]

model_1 = GradientBoostingRegressor(
    n_estimators=20,
    learning_rate=0.1,
    max_depth=2,
    subsample=0.8,
    random_state=42
)

pipe_gb = Pipeline([
    ("scaler", StandardScaler()),
    ("model", model_1)
])

print("\n--- 🚀 Training GradientBoosting ULTRAFAST ---")
pipe_gb.fit(X_train_gb, y_train_gb)
pred = pipe_gb.predict(X_test)
print("R2:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))
joblib.dump(pipe_gb, "GB_ultrafast.joblib")


# === ExtraTrees (fast) ===
model_2 = ExtraTreesRegressor(
    n_estimators=80,
    random_state=42,
    n_jobs=-1
)

pipe_et = Pipeline([
    ("scaler", StandardScaler()),
    ("model", model_2)
])

print("\n--- 🚀 Training ExtraTrees ULTRAFAST ---")
pipe_et.fit(X_train, y_train)
pred2 = pipe_et.predict(X_test)
print("R2:", r2_score(y_test, pred2))
print("MAE:", mean_absolute_error(y_test, pred2))
joblib.dump(pipe_et, "ET_ultrafast.joblib")

print("\n=== Done ===")
