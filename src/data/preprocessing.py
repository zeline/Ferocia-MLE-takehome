
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# Define feature groups
NUMERIC_FEATURES = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'day']
CATEGORICAL_FEATURES = ['job', 'marital', 'contact', 'poutcome']
BINARY_FEATURES = ['default', 'housing', 'loan']
ORDINAL_FEATURES = ['education', 'month']
TARGET = 'y'

#EXCLUDED FEATURES
EXCLUDE_FEATURES = []

#loading in
def load_data(filepath: str) -> pd.DataFrame:
    """
    load the dataset from a CSV file
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    #categorical missing values will be ignores
    #they make a minority unlike in poutcome
    missing_values = ["unknown"]
    df = pd.read_csv(filepath, sep=';', na_values = missing_values)
    return df

# ---- data transformations ----
def campaign_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add new campaign feature from pdays."""
    df = df.copy()
    # New binary feature: is this the first campaign for this client?
    df['first_campaign'] = (df['pdays'] == -1).astype(int)
    # Replace -1 with 0 for pdays (will be scaled anyway)
    df['pdays'] = df['pdays'].replace(-1, 0)
    # Only append if not already present (avoid duplicates)
    if 'first_campaign' not in BINARY_FEATURES:
        BINARY_FEATURES.append('first_campaign')
    return df

def exclude_features(df: pd.DataFrame) -> pd.DataFrame:
    #remove excluded features, errors='ignore' to avoid errors if excluded features are not present
    df = df.drop(EXCLUDE_FEATURES, axis=1, errors='ignore')
    return df
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
     #exclude future leaky features
    df = exclude_features(df)
    #drop target and return, ignore errors if target is not present
    df = df.drop('y', axis=1, errors='ignore')
    return df
def encode_target(y: pd.Series) -> np.ndarray:
    """
    Encode the target variable to binary.
    
    Args:
        y: Target variable with 'yes'/'no' values
        
    Returns:
        Binary encoded array (1 for 'yes', 0 for 'no')
    """
    return (y == 'yes').astype(int).values

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # add in what else i'd do
    # campaign feature transformation
    df = campaign_features(df)
    return df

#column transformers
def create_preprocessor() -> ColumnTransformer:
    """
    Create a preprocessing pipeline for the mixed data.
    
    Includes imputation for handling missing values created by treating 'unknown' as NaN.
    """
    
    # Numeric: impute with median (robust to outliers), then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical: impute with 'missing' category, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Ordinal: impute with most frequent, then encode
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    
    # Binary: impute with most frequent, then encode
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('ord', ordinal_transformer, ORDINAL_FEATURES),
            ('bin', binary_transformer, BINARY_FEATURES),
        ])
    return preprocessor

def split_data(X: pd.DataFrame, y: pd.Series, \
    test_size: float = 0.2, random_state: int = 77) -> tuple:
    """
    Split data into training and test sets with stratification.
    
    Args:
        X: Feature DataFrame
        y: Target array
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, \
    random_state=random_state, stratify=y)
# --- saving and loading to disk ---
def save_preprocessor(preprocessor: ColumnTransformer, filepath: str) -> None:
    """
    Save the fitted preprocessor to disk.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        filepath: Path to save the preprocessor
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, filepath)
def load_preprocessor(filepath: str) -> ColumnTransformer:
    """
    Load a fitted preprocessor from disk.
    
    Args:
        filepath: Path to the saved preprocessor
        
    Returns:
        Fitted ColumnTransformer
    """
    return joblib.load(filepath)

#the main function for training
def preprocess_for_training(filepath: str) -> tuple:
    """
    Complete preprocessing pipeline for training.
    
    Args:
        filepath: Path to the dataset CSV
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, fitted_preprocessor)
    """
    # Load and clean data
    df = load_data(filepath)
    df = clean_data(df)
    
    # Separate features and target
    X = prepare_features(df)
    y = encode_target(df[TARGET])
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor()
        #fit transform to fit the model
    X_train_processed = preprocessor.fit_transform(X_train)
        #transform to apply the learned parameters on unseen test data
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

#for future API calls
def preprocess_for_inference(data: dict, preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Preprocess a single sample or batch for inference.
    
    Args:
        data: Dictionary or list of dictionaries with feature values
        preprocessor: Fitted ColumnTransformer
        
    Returns:
        Preprocessed feature array
    """
    # Convert to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    
    # Apply cleaning
    df = clean_data(df)
    
    # Prepare features
    X = prepare_features(df)
    
    # Transform
    return preprocessor.transform(X)