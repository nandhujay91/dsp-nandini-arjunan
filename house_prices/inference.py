import os
import joblib
import pandas as pd
import numpy as np

# Define paths
MODEL_FOLDER_PATH = '../models'


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Makes predictions using a pre-trained linear regression model.

    Args:
        input_data (pd.DataFrame): The input DataFrame
        containing the features for prediction.

    Returns:
        np.ndarray: The predicted values.
    """
    # Load objects
    scaler = joblib.load(os.path.join(MODEL_FOLDER_PATH, 'scaler.joblib'))
    one_hot_encoder = joblib.load(
        os.path.join(MODEL_FOLDER_PATH, 'one_hot_encoder.joblib')
        )
    model = joblib.load(os.path.join(MODEL_FOLDER_PATH, 'model.joblib'))

    # Define constants
    CONTINUOUS_FEATURES = ['TotRmsAbvGrd', 'YrSold', '1stFlrSF', 'WoodDeckSF']
    FEATURES_TO_ONE_HOT_ENCODE = ['Foundation']
    KITCHEN_QUALITY_DICT = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1}

    # Process input data
    X_input_one_hot_encoded = one_hot_encoder.transform(
        input_data[FEATURES_TO_ONE_HOT_ENCODE]
        )
    one_hot_encoded_feature_names = one_hot_encoder.get_feature_names_out(
        FEATURES_TO_ONE_HOT_ENCODE
        )
    X_input_one_hot_encoded_df = pd.DataFrame(
        X_input_one_hot_encoded, columns=one_hot_encoded_feature_names
        )

    X_kitchen_quality_encode = input_data['KitchenQual'].apply(
        lambda x: KITCHEN_QUALITY_DICT.get(x, 0)
        )
    X_input_kitchen_quality_encoded_df = X_kitchen_quality_encode.to_frame(
        name='KitchenQual_encoded'
        )

    X_input_scaled = scaler.transform(input_data[CONTINUOUS_FEATURES])
    X_input_scaled_df = pd.DataFrame(
        X_input_scaled, columns=CONTINUOUS_FEATURES
        )

    X_input_processed = pd.concat(
        [X_input_one_hot_encoded_df, X_input_kitchen_quality_encoded_df,
         X_input_scaled_df], axis=1
        )

    # Make predictions
    return model.predict(X_input_processed)
