import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def encode_kitchen_quality(df: pd.DataFrame, column: str) -> pd.Series:
    # Encode the kitchen quality column to numerical values.
    kitchen_quality_dict = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1}
    return df[column].apply(lambda x: kitchen_quality_dict[x])


def preprocess_data(
    df: pd.DataFrame,
    one_hot_encoder: OneHotEncoder,
    scaler: StandardScaler
) -> pd.DataFrame:
    # Preprocess the input DataFrame by applying one-hot encoding and scaling.
    # One-hot encode the 'Foundation' column
    foundation_encoded = pd.DataFrame(
        one_hot_encoder.transform(df[['Foundation']]),
        columns=one_hot_encoder.get_feature_names_out())
    # Encode the 'KitchenQual' column
    kitchen_quality_encoded = encode_kitchen_quality(
        df,
        'KitchenQual'
    ).rename('KitchenQual')
    # Scale continuous features
    continuous_features = [
        'TotRmsAbvGrd', 'YrSold', 'WoodDeckSF', '1stFlrSF']
    scaled_data = pd.DataFrame(
        scaler.transform(df[continuous_features]),
        columns=continuous_features
    )
    # Concatenate all processed features
    processed_data = pd.concat(
        [foundation_encoded, kitchen_quality_encoded, scaled_data], axis=1)
    return processed_data


def fit_preprocessors(df: pd.DataFrame):
    # Initialize the preprocessors
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    scaler = StandardScaler()
    # Fit the one-hot encoder on the 'Foundation' column
    one_hot_encoder.fit(df[['Foundation']])
    # Fit the scaler on the continuous features
    continuous_features = ['TotRmsAbvGrd', 'YrSold', 'WoodDeckSF', '1stFlrSF']
    scaler.fit(df[continuous_features])
    return one_hot_encoder, scaler
