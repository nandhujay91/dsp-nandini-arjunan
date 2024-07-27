import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.impute import SimpleImputer
from typing import Dict
# Define paths
MODEL_FOLDER_PATH = '../models'


def build_model(data: pd.DataFrame) -> Dict[str, float]:
    """
    Builds and trains a linear regression model using the provided dataset.

    Args:
        data (pd.DataFrame): The input DataFrame containing features
          and the target column 'SalePrice'.

    Returns:
        Dict[str, float]: A dictionary containing the
        Root Mean Squared Error (RMSE) of the model on the test set.
    """
    # Constants
    LABEL_COL = 'SalePrice'
    USEFUL_FEATURES = ['Foundation', 'KitchenQual', 'TotRmsAbvGrd', 'WoodDeckSF', 'YrSold', '1stFlrSF']
    CONTINUOUS_FEATURES = ['TotRmsAbvGrd', 'YrSold', '1stFlrSF', 'WoodDeckSF']
    FEATURES_TO_ONE_HOT_ENCODE = ['Foundation']
    KITCHEN_QUALITY_DICT = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1}

    # Split data
    X = data[USEFUL_FEATURES]
    y = data[LABEL_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Prepare encoders
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoder.fit(X_train[FEATURES_TO_ONE_HOT_ENCODE])

    scaler = StandardScaler()
    scaler.fit(X_train[CONTINUOUS_FEATURES])

    # Process train data
    X_train_one_hot_encoded = one_hot_encoder.transform(X_train[FEATURES_TO_ONE_HOT_ENCODE])
    one_hot_encoded_feature_names = one_hot_encoder.get_feature_names_out(FEATURES_TO_ONE_HOT_ENCODE)
    X_train_one_hot_encoded_df = pd.DataFrame(X_train_one_hot_encoded, columns=one_hot_encoded_feature_names)

    X_train_kitchen_quality_encoded = X_train['KitchenQual'].map(KITCHEN_QUALITY_DICT).fillna(0)
    X_train_kitchen_quality_encoded_df = X_train_kitchen_quality_encoded.to_frame(name='KitchenQual_encoded')

    X_train_scaled = scaler.transform(X_train[CONTINUOUS_FEATURES])
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=CONTINUOUS_FEATURES)

    # Reset index before concatenation
    X_train_one_hot_encoded_df = X_train_one_hot_encoded_df.reset_index(drop=True)
    X_train_kitchen_quality_encoded_df = X_train_kitchen_quality_encoded_df.reset_index(drop=True)
    X_train_scaled_df = X_train_scaled_df.reset_index(drop=True)

    X_train_processed = pd.concat([X_train_one_hot_encoded_df, X_train_kitchen_quality_encoded_df, X_train_scaled_df], axis=1)

    # Handle missing values if any
    if X_train_processed.isna().sum().sum() > 0:
        imputer = SimpleImputer(strategy='mean')
        X_train_processed = imputer.fit_transform(X_train_processed)

    # Train model
    model = LinearRegression()
    model.fit(X_train_processed, y_train)

    # Save objects
    joblib.dump(scaler, os.path.join(MODEL_FOLDER_PATH, 'scaler.joblib'))
    joblib.dump(one_hot_encoder, os.path.join(MODEL_FOLDER_PATH, 'one_hot_encoder.joblib'))
    joblib.dump(model, os.path.join(MODEL_FOLDER_PATH, 'model.joblib'))

    # Process test data
    X_test_one_hot_encoded = one_hot_encoder.transform(X_test[FEATURES_TO_ONE_HOT_ENCODE])
    X_test_one_hot_encoded_df = pd.DataFrame(X_test_one_hot_encoded, columns=one_hot_encoded_feature_names)

    X_test_kitchen_quality_encoded = X_test['KitchenQual'].map(KITCHEN_QUALITY_DICT).fillna(0)
    X_test_kitchen_quality_encoded_df = X_test_kitchen_quality_encoded.to_frame(name='KitchenQual_encoded')

    X_test_scaled = scaler.transform(X_test[CONTINUOUS_FEATURES])
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=CONTINUOUS_FEATURES)

    # Reset index before concatenation
    X_test_one_hot_encoded_df = X_test_one_hot_encoded_df.reset_index(drop=True)
    X_test_kitchen_quality_encoded_df = X_test_kitchen_quality_encoded_df.reset_index(drop=True)
    X_test_scaled_df = X_test_scaled_df.reset_index(drop=True)

    X_test_processed = pd.concat([X_test_one_hot_encoded_df, X_test_kitchen_quality_encoded_df, X_test_scaled_df], axis=1)

    # Handle missing values if any
    if X_test_processed.isna().sum().sum() > 0:
        X_test_processed = imputer.transform(X_test_processed)

    # Predict and evaluate
    y_pred = model.predict(X_test_processed)

    rmse = float(root_mean_squared_error(y_test, y_pred))

    return {'rmse': rmse}
