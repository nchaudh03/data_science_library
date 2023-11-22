from abc import ABC, abstractmethod
import optuna
from abc import ABC, abstractmethod
import optuna

class HyperparameterSearchProtocol(ABC):
    """
    A protocol for hyperparameter search using Optuna.

    This protocol defines the structure for implementing hyperparameter search
    algorithms using the Optuna library. Subclasses must implement the `objective`
    method, which defines the objective function to be optimized.

    Attributes:
        None

    Methods:
        objective(trial: optuna.Trial) -> float:
            Abstract method that defines the objective function to be optimized.
            This method takes an Optuna Trial object as input and returns a float
            representing the objective value.

        optimize(study_name: str, n_trials: int) -> optuna.study.Study:
            Optimize the objective function using the Optuna library.
            This method creates an Optuna study, performs the optimization using
            the defined objective function, and returns the resulting study object.

    Usage:
        To use this protocol, create a subclass that implements the `objective`
        method. Then, instantiate the subclass and call the `optimize` method
        to perform the hyperparameter search.

    Example:
        class MyHyperparameterSearch(HyperparameterSearchProtocol):
            def objective(self, trial: optuna.Trial) -> float:
                # Define the objective function to be optimized
                ...

        search = MyHyperparameterSearch()
        study = search.optimize("my_study", 100)
    """
    @abstractmethod
    def objective(self, trial: optuna.Trial) -> float:
        pass

    def optimize(self, study_name: str, n_trials: int) -> optuna.study.Study:
        study = optuna.create_study(study_name=study_name)
        study.optimize(self.objective, n_trials=n_trials)
        return study



