from typing import Any, Tuple, Union

import pandas as pd
from sklearn.datasets import make_classification, make_regression

from data_science_library.src.protocals.dataloader import DataLoaderProtocol


class SampleLoader(DataLoaderProtocol):
    def __init__(self, sample_type: str) -> None:
        self.sample_type: str = sample_type

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.sample_type == "regression":
            X, y = make_regression(n_samples=100, n_features=10)
        elif self.sample_type == "binary":
            X, y = make_classification(
                n_samples=100, n_features=10, n_informative=2, n_classes=2
            )
        elif self.sample_type == "multiclass":
            X, y = make_classification(
                n_samples=100, n_features=10, n_informative=4, n_classes=3
            )
        else:
            raise ValueError("Invalid sample type")

        X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        y = pd.DataFrame(y, columns=["y"])

        return X, y
