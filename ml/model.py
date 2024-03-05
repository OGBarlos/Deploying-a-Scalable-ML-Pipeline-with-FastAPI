import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import f1_score, precision_score, recall_score
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier
import joblib
from joblib import dump
import os

import multiprocessing

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    fbeta = f1_score(y, preds)
    
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    
    return preds

def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    joblib.dump(model, path)

def load_model(path):
    """
    Loads model and related objects from the given path.

    Inputs
    ------
    path : str
        Path where model and related objects are saved.
    Returns
    -------
    model : object
        Trained machine learning model.
    encoder : object
        Trained OneHotEncoder.
    lb : object
        Trained LabelBinarizer.
    """
    logging.info(f"Loading model from: {path}")
    model = joblib.load(os.path.join(path, "model.joblib"))
    encoder = joblib.load(os.path.join(path, "encoder.joblib"))
    lb = joblib.load(os.path.join(path, "lb.joblib"))
    return model, encoder, lb
    

def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # TODO: implement the function
    sliced_data = data[data[column_name] == slice_value].copy()

    # Extract features and labels
    X_slice = sliced_data.drop(columns=[label])
    y_slice = sliced_data[label] if label else None

    # One hot encode categorical features
    if categorical_features and encoder is not None:
        X_encoded = encoder.transform(X_slice[categorical_features])
    else:
        X_encoded = np.array([])

    # Binarize labels
    if label is not None and lb is not None:
        y_binarized = lb.transform(y_slice)
    else:
        y_binarized = np.array([])

    # Make predictions
    if model is not None:
        preds = model.predict(X_encoded)
    else:
        raise ValueError("Model is not provided.")

    # Compute precision, recall, and F1 scores
    precision = precision_score(y_binarized, preds)
    recall = recall_score(y_binarized, preds)
    fbeta = f1_score(y_binarized, preds)
    
    return precision, recall, fbeta