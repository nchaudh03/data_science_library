from typing import Any, Dict, Tuple

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error

from data_science_library.src.protocals.hyperparameter_search import (
    HyperparameterSearchProtocol,
)


class LightGBMHyperparameterSearch(HyperparameterSearchProtocol):
    """
    A class for performing hyperparameter search for LightGBM models.

    Example usage:
    ```
    train_data = (train_features, train_labels)
    valid_data = (valid_features, valid_labels)
    params = {'num_leaves': 32, 'learning_rate': 0.1, 'feature_fraction': 0.8}
    task = 'classification'

    hyperparameter_search = LightGBMHyperparameterSearch(train_data, valid_data, params, task)
    best_params = hyperparameter_search.run()
    ```
    """

    def __init__(
        self,
        train_data: Tuple[pd.DataFrame, pd.DataFrame],
        valid_data: Tuple[pd.DataFrame, pd.DataFrame],
        params: Dict[str, Any],
        task: str,
    ):
        """
        Initialize the LightGBMHyperparameterSearch instance.

        Args:
            train_data: A tuple containing the training data (features, labels).
            valid_data: A tuple containing the validation data (features, labels).
            params: A dictionary containing the initial hyperparameter values.
            task: A string indicating the task type ('classification' or 'regression').
        """
        self.train_data = train_data
        self.valid_data = valid_data
        self.params = params
        self.task = task

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for the hyperparameter search.

        Args:
            trial: An Optuna Trial object for generating hyperparameter suggestions.

        Returns:
            The loss value (log loss or mean squared error) for the current set of hyperparameters.
        """
        param = self.params.copy()
        param.update(
            {
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
                "feature_fraction": trial.suggest_uniform("feature_fraction", 0.1, 1.0),
                "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.1, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }
        )

        print(param)

        train_x, train_y = self.train_data
        valid_x, valid_y = self.valid_data

        train_dataset = lgb.Dataset(train_x, label=train_y)
        valid_dataset = lgb.Dataset(valid_x, label=valid_y)

        model = lgb.train(param, train_dataset, valid_sets=[valid_dataset])
        predictions = model.predict(valid_x)

        if self.task == "binary":
            loss_value = log_loss(valid_y, predictions)
        elif self.task == "regression":
            loss_value = mean_squared_error(valid_y, predictions)
        elif self.task == "multiclass":
            loss_value = log_loss(
                valid_y, predictions
            )  # You can modify this accordingly for multiclass loss
        else:
            raise ValueError(
                "Invalid task type. Supported task types are 'binary', 'regression', and 'multiclass'."
            )

        return loss_value
