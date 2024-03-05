import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, load_model, compute_model_metrics, performance_on_categorical_slice

# Load the census data
project_path = os.getcwd()
data_path = os.path.join(project_path, "data", "census.csv")

data = pd.read_csv(data_path)

# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Split the data into train and test sets
train, test = train_test_split(data, test_size=0.20, random_state=10, stratify=data['salary'])

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True  # Corrected from is_training=True
)

# Process the testing data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,  # Corrected from is_training=False
    encoder=encoder,
    lb=lb
)

# Train the model
model = train_model(X_train, y_train)

# Save the model
model_dir = os.path.join(project_path, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "model.pkl")
save_model(model, model_path)

# Load the model
model = load_model(model_path)

# Run inference on the test set
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute the performance on model slices using the performance_on_categorical_slice function
# Iterate through the categorical features
for col in cat_features:
    # Iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            test, col, slicevalue, cat_features, "salary", encoder, lb, model
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)

