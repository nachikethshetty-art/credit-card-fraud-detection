import joblib
import pandas as pd
import os

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_rf_model.pkl")
INPUT_CSV = os.path.join(BASE_DIR, "data", "sample_transactions.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "scored_transactions.csv")

THRESHOLD = 0.3

# -----------------------------
# Load model
# -----------------------------
print("🔹 Loading model...")
model = joblib.load(MODEL_PATH)

# -----------------------------
# Load input data
# -----------------------------
print("🔹 Reading input CSV...")
df_input = pd.read_csv(INPUT_CSV)

print(f"Input shape: {df_input.shape}")

# -----------------------------
# Predict
# -----------------------------
print("🔹 Running fraud predictions...")

probs = model.predict_proba(df_input)[:, 1]
preds = (probs >= THRESHOLD).astype(int)

# -----------------------------
# Append results
# -----------------------------
df_output = df_input.copy()
df_output["fraud_probability"] = probs
df_output["fraud_prediction"] = preds

# -----------------------------
# Save results
# -----------------------------
df_output.to_csv(OUTPUT_CSV, index=False)

print("✅ Scoring complete!")
print(f"📁 Results saved to: {OUTPUT_CSV}")