import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.compose import TransformedTargetRegressor


# Find the CSV in the same folder as this Python file
file_path = Path(__file__).parent / "pakistan_cars_combined.csv"

# Load the CSV into a table
data = pd.read_csv(file_path, low_memory=False)

# Inspect the dataset
print("Total rows:", data.shape[0])
print("Total columns:", data.shape[1])

print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst five cars:")
print(
    data[
        ["make", "model", "year", "km_driven", "price_pkr"]
    ].head().to_string(index=False)
)
# Show the problems identified during preparation
print("\nQuality-check results:")
print(data["quality_flags"].value_counts())

# Keep only records that passed the basic checks
data = data[
    data["basic_training_eligible"] == True
].copy()

print("\nRows after filtering:", len(data))

# Inspect the inputs we plan to use
features = [
    "make",
    "model",
    "year",
    "km_driven",
    "engine_cc",
    "fuel_type",
    "transmission",
    "city"
]

print("\nMissing values in these inputs:")
print(data[features].isnull().sum())

print("\nMost common make/model combinations:")
print(
    data.groupby(["make", "model"])
    .size()
    .sort_values(ascending=False)
    .head(15)
)


from sklearn.model_selection import GroupShuffleSplit

# Inputs used to predict the price
X = data[features]

# Correct prices: the target
y = data["price_pkr"]

# Similar records must stay together
groups = data["possible_duplicate_group"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Confirm that no group appears in both sets
train_groups = set(groups.iloc[train_index])
test_groups = set(groups.iloc[test_index])

overlap = train_groups.intersection(test_groups)

print("Groups shared between training and testing:", len(overlap))


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Separate numerical and categorical inputs
numeric_features = ["year", "km_driven", "engine_cc"]

categorical_features = [
    "make",
    "model",
    "fuel_type",
    "transmission",
    "city"
]

# Prepare numerical inputs
numeric_pipeline = Pipeline([
    ("fill_missing", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

# Prepare categorical inputs
categorical_pipeline = Pipeline([
    (
        "fill_missing",
        SimpleImputer(
            strategy="constant",
            fill_value="UNKNOWN"
        )
    ),
    (
        "encode",
        OneHotEncoder(handle_unknown="ignore")
    )
])

# Apply each preparation to the appropriate columns
preprocessor = ColumnTransformer([
    ("numbers", numeric_pipeline, numeric_features),
    ("categories", categorical_pipeline, categorical_features)
])

# Connect preparation and prediction
# Prepare inputs and learn the logarithm of price
price_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model = TransformedTargetRegressor(
    regressor=price_pipeline,
    func=np.log,
    inverse_func=np.exp
)
# Automatically transform prices during training
# and convert predictions back to PKR


print("\nTraining the model...")
model.fit(X_train, y_train)

print("Pakistan car-price model trained successfully!")


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Predict prices for the testing records
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"\nMAE: PKR {mae:,.0f}")
print(f"MAE in lakh: {mae / 100000:.2f}")

print(f"RMSE: PKR {rmse:,.0f}")
print(f"R² score: {r2:.3f}")

# Check for impossible price predictions
print("Zero or negative predictions:", (y_pred <= 0).sum())

# Compare actual and predicted prices
comparison = X_test[["make", "model", "year"]].copy()
comparison["actual_price_pkr"] = y_test
comparison["predicted_price_pkr"] = y_pred.round(0)

print("\nFirst 10 predictions:")
print(comparison.head(10).to_string(index=False))


# Inspect training prices only
print("\nTraining price distribution:")
print(
    y_train.describe(
        percentiles=[0.50, 0.90, 0.95, 0.99]
    ).round(0)
)

# Display the most expensive training records
price_check = X_train.copy()
price_check["price_pkr"] = y_train

print("\n10 most expensive training records:")
print(
    price_check.nlargest(10, "price_pkr")[
        ["make", "model", "year", "price_pkr"]
    ].to_string(index=False)
)


# Combine car details with actual and predicted prices
results = X_test[["make", "model"]].copy()

results["actual_price"] = y_test
results["predicted_price"] = y_pred

results["absolute_error"] = (
    results["actual_price"] - results["predicted_price"]
).abs()

# Summarize errors for each make/model combination
model_report = (
    results.groupby(["make", "model"])
    .agg(
        test_cars=("absolute_error", "size"),
        mae_pkr=("absolute_error", "mean"),
        median_error_pkr=("absolute_error", "median")
    )
    .sort_values("test_cars", ascending=False)
)

model_report["mae_lakh"] = model_report["mae_pkr"] / 100000

print("\nPerformance on the 15 most common test models:")
print(model_report.head(15).round(2).to_string())



import joblib

folder = Path(__file__).parent

# Save the complete model, including the log-price transformation
joblib.dump(
    model,
    folder / "pakistan_car_price_model.pkl"
)

# Save training input combinations for the future app's dropdowns
X_train.drop_duplicates().to_csv(
    folder / "pakistan_car_options.csv",
    index=False
)

# Save the evaluation table for reference
model_report.to_csv(
    folder / "pakistan_model_report.csv"
)

print("\nModel, input options, and evaluation report saved!")



