import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# File names to save and load the trained model and pipeline
MODEL_FILE = "model.pkl"
PIPELINE_FILE = 'pipeline.pkl'

# ===========================
# Function to build preprocessing pipeline
# ===========================
def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # Replace missing numeric values with median
        ("scaler", StandardScaler())                   # Standardize numeric features
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))  # Convert categories into one-hot vectors
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    return full_pipeline

# ===========================
# Training mode (if model does not exist)
# ===========================
if not os.path.exists(MODEL_FILE):
    # Load the dataset
    housing = pd.read_csv("housing.csv")

    # Create an income category column for stratified sampling
    housing['income_cat'] = pd.cut(housing["median_income"],
                                   bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                   labels=[1, 2, 3, 4, 5])

    # Split the data into train and test sets using stratified sampling
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        # Save the test features and labels separately
        test_features = housing.loc[test_index].drop(["median_house_value", "income_cat"], axis=1)
        test_labels = housing.loc[test_index, "median_house_value"]
        test_features.to_csv("input.csv", index=False)
        test_labels.to_csv("test_labels.csv", index=False)

        # Keep the training data (drop only income_cat)
        housing = housing.loc[train_index].drop("income_cat", axis=1)

    # Separate features and labels for training
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    # Identify numerical and categorical attributes
    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    # Build the pipeline
    pipeline = build_pipeline(num_attribs, cat_attribs)

    # Preprocess the training data
    housing_prepared = pipeline.fit_transform(housing_features)

    # Train the Random Forest model
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model.fit(housing_prepared, housing_labels)

    # ===========================
    # Evaluate training and test accuracy
    # ===========================
    # Training accuracy
    train_predictions = model.predict(housing_prepared)
    train_accuracy = r2_score(housing_labels, train_predictions) * 100

    # Test accuracy
    test_features = pd.read_csv("input.csv")
    test_labels = pd.read_csv("test_labels.csv")
    test_prepared = pipeline.transform(test_features)
    test_predictions = model.predict(test_prepared)
    test_accuracy = r2_score(test_labels, test_predictions) * 100

    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Save the model and pipeline to disk
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is trained and saved!")

# ===========================
# Inference mode (if model already exists)
# ===========================
else:
    # Load model and pipeline
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    # Load the new input data (test set)
    input_data = pd.read_csv('input.csv')

    # Preprocess the input data using the saved pipeline
    transformed_input = pipeline.transform(input_data)

    # Make predictions
    predictions = model.predict(transformed_input)

    # Add predictions as a new column
    input_data['median_house_value'] = predictions

    # Save the results to output.csv
    input_data.to_csv("output.csv", index=False)
    print("Inference complete, results saved to output.csv")
