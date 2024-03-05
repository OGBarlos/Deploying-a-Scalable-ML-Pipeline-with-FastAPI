import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """
    if training:
        # One hot encode categorical features
        if categorical_features:
            encoder = OneHotEncoder(sparse=False)
            X_encoded = encoder.fit_transform(X[categorical_features])
        else:
            X_encoded = np.array([])

        # Binarize labels
        if label is not None:
            lb = LabelBinarizer()
            y = lb.fit_transform(X[label])
        else:
            y = np.array([])

        return X_encoded, y, encoder, lb
    else:
        # One hot encode categorical features
        if categorical_features and encoder is not None:
            X_encoded = encoder.transform(X[categorical_features])
        else:
            X_encoded = np.array([])

        # Binarize labels
        if label is not None and lb is not None:
            y = lb.transform(X[label])
        else:
            y = np.array([])

        return X_encoded, y, encoder, lb

def apply_label(inference):
    """ Convert the binary label in a single inference sample into string output."""
    if inference[0] == 1:
        return ">50K"
    elif inference[0] == 0:
        return "<=50K"

