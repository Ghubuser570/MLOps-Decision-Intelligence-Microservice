import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from catboost import CatBoostClassifier

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

NUMERIC_FEATURES = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_FEATURES = ["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]

print("⏳ Siphoning dataset directly from the UCI Machine Learning matrix...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, names=COLUMNS, sep=r",\s*", engine="python", na_values="?")
df = df.dropna()

y = (df["income"].str.contains(">50K")).astype(int).values
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("⚙️ Fitting Preprocessor...")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ]
)
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)

print("🌲 Training LightGBM...")
lgb_train = lgb.Dataset(X_train_proc, label=y_train)
lgb_params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1}
lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=50)

print("🐱 Training CatBoost...")
cat_model = CatBoostClassifier(iterations=50, verbose=0, random_seed=42)
cat_model.fit(X_train_proc, y_train)

print("⚖️ Calibrating Ensemble...")
lgb_val_preds = lgb_model.predict(X_val_proc)
cat_val_preds = cat_model.predict_proba(X_val_proc)[:, 1]
ensemble_val_preds = (lgb_val_preds + cat_val_preds) / 2.0

calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(ensemble_val_preds, y_val)

os.makedirs("models", exist_ok=True)
lgb_model.save_model("models/lightgbm_model.bin")
cat_model.save_model("models/catboost_model.bin")
joblib.dump(preprocessor, "models/preprocessor.joblib")
joblib.dump(calibrator, "models/calibrator.joblib")

print("✅ All 4 native models exported successfully to models/!")