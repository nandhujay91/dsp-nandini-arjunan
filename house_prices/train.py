import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from typing import Dict
import numpy as np

# Define paths
MODEL_FOLDER_PATH = '../models'


def split_data(data: pd.DataFrame) -> tuple:
    USEFUL_FEATURES = [
        'Foundation', 'KitchenQual', 'TotRmsAbvGrd', 'WoodDeckSF', 'YrSold',
        '1stFlrSF'
    ]
    LABEL_COL = 'SalePrice'
    X = data[USEFUL_FEATURES]
    y = data[LABEL_COL]
    return train_test_split(X, y, test_size=0.33, random_state=42)


def prepare_encoders(X_train: pd.DataFrame) -> tuple:
    FEATURES_TO_ONE_HOT_ENCODE = ['Foundation']
    CONTINUOUS_FEATURES = [
        'TotRmsAbvGrd', 'YrSold', '1stFlrSF', 'WoodDeckSF'
    ]
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    scaler = StandardScaler()
    one_hot_encoder.fit(X_train[FEATURES_TO_ONE_HOT_ENCODE])
    scaler.fit(X_train[CONTINUOUS_FEATURES])
    return one_hot_encoder, scaler


def encode_features(
    X: pd.DataFrame, one_hot_encoder: OneHotEncoder,
    scaler: StandardScaler
) -> pd.DataFrame:
    CONTINUOUS_FEATURES = [
        'TotRmsAbvGrd', 'YrSold', '1stFlrSF', 'WoodDeckSF'
    ]
    FEATURES_TO_ONE_HOT_ENCODE = ['Foundation']
    KITCHEN_QUALITY_DICT = {
        'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1
    }
    X_one_hot_encoded = one_hot_encoder.transform(
        X[FEATURES_TO_ONE_HOT_ENCODE]
    )
    X_kitchen_quality_encoded = X['KitchenQual'].map(
        KITCHEN_QUALITY_DICT
    ).fillna(0).to_frame(name='KitchenQual_encoded')
    X_scaled = scaler.transform(X[CONTINUOUS_FEATURES])
    return pd.concat(
        [
            pd.DataFrame(
                X_one_hot_encoded,
                columns=one_hot_encoder.get_feature_names_out(
                    FEATURES_TO_ONE_HOT_ENCODE
                )
            ),
            X_kitchen_quality_encoded.reset_index(drop=True),
            pd.DataFrame(X_scaled, columns=CONTINUOUS_FEATURES).reset_index(
                drop=True
            )
        ],
        axis=1
    )


def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    if X.isna().sum().sum() > 0:
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    return pd.DataFrame(X)


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def save_objects(
    model: LinearRegression, scaler: StandardScaler,
    one_hot_encoder: OneHotEncoder
) -> None:
    joblib.dump(scaler, os.path.join(MODEL_FOLDER_PATH, 'scaler.joblib'))
    joblib.dump(
        one_hot_encoder,
        os.path.join(MODEL_FOLDER_PATH, 'one_hot_encoder.joblib')
    )
    joblib.dump(model, os.path.join(MODEL_FOLDER_PATH, 'model.joblib'))


def evaluate_model(
    model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {'rmse': rmse}


def build_model(data: pd.DataFrame) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = split_data(data)
    one_hot_encoder, scaler = prepare_encoders(X_train)
    X_train_processed = encode_features(X_train, one_hot_encoder, scaler)
    X_train_processed = handle_missing_values(X_train_processed)
    model = train_model(X_train_processed, y_train)
    save_objects(model, scaler, one_hot_encoder)
    X_test_processed = encode_features(X_test, one_hot_encoder, scaler)
    X_test_processed = handle_missing_values(X_test_processed)
    return evaluate_model(model, X_test_processed, y_test)
