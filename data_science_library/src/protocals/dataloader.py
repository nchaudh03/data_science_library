from abc import ABC, abstractmethod
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class DataLoaderProtocol(ABC):
    """
    Abstract base class for data loader protocols.

    This class provides a protocol for loading and preprocessing data, as well as splitting it into train and test sets.
    Subclasses must implement the `load_data` method to load the raw data.

    Attributes:
        X_train (np.ndarray): The preprocessed training features.
        X_test (np.ndarray): The preprocessed test features.
        y_train (pd.Series): The training target variable.
        y_test (pd.Series): The test target variable.
        preprocessor (ColumnTransformer): The data preprocessor.

    Methods:
        load_data: Abstract method to load the raw data.
        preprocess_data: Preprocesses the data by applying transformations to numerical, binary, and categorical variables.
        prepare_data: Loads and preprocesses the data, and splits it into train and test sets.
        get_train_data: Returns the training data.
        get_test_data: Returns the test data.
        get_test_labels: Returns the test labels.
    """

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        
    @abstractmethod
    def load_data(self):
        """
        Abstract method to load the raw data.

        Subclasses must implement this method to load the raw data.

        Returns:
            tuple: A tuple containing the input features (X) and the target variable (y).
        """
        pass
    
    def preprocess_data(self, X, y):
        """
        Preprocesses the data by applying transformations to numerical, binary, and categorical variables.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.

        Returns:
            X_preprocessed (np.ndarray): The preprocessed features.
            y (pd.Series): The target variable.
        """
        # Preprocessing for numerical variables
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Preprocessing for binary variables
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('labeler', LabelEncoder())
        ])

        # Preprocessing for categorical variables
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder())
        ])

        # Combine all preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, X.select_dtypes(include=['float64', 'int64']).columns),
                ('bin', binary_transformer, X.select_dtypes(include=['bool']).columns),
                ('cat', categorical_transformer, X.select_dtypes(include=['object']).columns)
            ])

        # Apply preprocessing to X
        X_preprocessed = pd.DataFrame(self.preprocessor.fit_transform(X), columns=self.preprocessor.get_feature_names_out())

        return X_preprocessed, y

    def prepare_data(self):
        """
        Loads and preprocesses the data, and splits it into train and test sets.

        Args:
            problem_type (str): regression, binary, multiclassification.

        Returns:
            X_train (np.ndarray): The preprocessed training features.
            X_test (np.ndarray): The preprocessed test features.
            y_train (pd.Series): The training target variable.
            y_test (pd.Series): The test target variable.
        """
        X, y = self.load_data()
        X_preprocessed, y = self.preprocess_data(X, y)

        # Split the dataset into train and test sets
        if self.sample_type != "regression":
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_preprocessed, y, test_size=0.2, stratify=y)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_preprocessed, y, test_size=0.2)

        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_train_data(self):
        """
        Returns the training data.

        Returns:
            tuple: A tuple containing the training features (X_train) and the training target variable (y_train).
        """
        return (self.X_train, self.y_train)

    def get_test_data(self):
        """
        Returns the test data.

        Returns:
            np.ndarray: The test features (X_test).
        """
        return self.X_test

    def get_test_labels(self):
        """
        Returns the test labels.

        Returns:
            pd.Series: The test target variable (y_test).
        """
        return self.y_test
