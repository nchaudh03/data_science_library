import xgboost as xgb
from typing import Any, Dict, Tuple
from sklearn.metrics import log_loss, mean_squared_error
from data_science_library.src.protocals.hyperparameter_search import HyperparameterSearchProtocol
import optuna
import pandas as pd


class XGBoostHyperparameterSearch(HyperparameterSearchProtocol):
    """
    Class for performing hyperparameter search for XGBoost models.

    Args:
        train_data (Tuple[pd.DataFrame, pd.DataFrame]): Tuple containing the training data features and labels.
        valid_data (Tuple[pd.DataFrame, pd.DataFrame]): Tuple containing the validation data features and labels.
        params (Dict[str, Any]): Dictionary containing the XGBoost hyperparameters.
        task (str): The task type, either 'classification' or 'regression'.

    Attributes:
        train_data (Tuple[pd.DataFrame, pd.DataFrame]): Tuple containing the training data features and labels.
        valid_data (Tuple[pd.DataFrame, pd.DataFrame]): Tuple containing the validation data features and labels.
        params (Dict[str, Any]): Dictionary containing the XGBoost hyperparameters.
        task (str): The task type, either 'classification' or 'regression'.
    """

    def __init__(self, train_data: Tuple[pd.DataFrame, pd.DataFrame], valid_data: Tuple[pd.DataFrame, pd.DataFrame], params: Dict[str, Any], task: str):
        self.train_data = train_data
        self.valid_data = valid_data
        self.params = params
        self.task = task

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for the hyperparameter search.

        Args:
            trial (optuna.Trial): The Optuna trial object.

        Returns:
            float: The evaluation metric value for the current set of hyperparameters.
        """
        param = {

            'booster': 'gbtree',
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_loguniform('gamma', 0.01, 1.0),  # Additional hyperparameter
            'lambda': trial.suggest_loguniform('lambda', 0.01, 1.0),  # Additional hyperparameter
        }
        
        param.update(self.params)

        train_x, train_y = self.train_data
        valid_x, valid_y = self.valid_data

        train_dataset = xgb.DMatrix(train_x, label=train_y)
        valid_dataset = xgb.DMatrix(valid_x, label=valid_y)

        model = xgb.train(param, train_dataset, evals=[(valid_dataset, 'validation')], early_stopping_rounds=10)
        predictions = model.predict(valid_dataset)

        return log_loss(valid_y, predictions) if self.task == 'classification' else mean_squared_error(valid_y, predictions)
