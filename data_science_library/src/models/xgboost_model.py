from data_science_library.src.protocals.dataloader import DataLoaderProtocol
from data_science_library.src.protocals.models import ModelsProtocol
import xgboost as xgb

class XGBoostModel(ModelsProtocol):
    """
    XGBoostModel is a class that represents a model trained using the XGBoost algorithm.

    Parameters:
    - params (dict): A dictionary containing the hyperparameters for the XGBoost model.

    Attributes:
    - params (dict): A dictionary containing the hyperparameters for the XGBoost model.
    - model: The trained XGBoost model.

    Methods:
    - train(dataloader: DataLoaderProtocol): Trains the XGBoost model using the provided dataloader.
    - predict(dataloader: DataLoaderProtocol): Makes predictions using the trained XGBoost model on the test data from the dataloader.
    """

    def __init__(self, params):
        """
        Initializes a new instance of the XGBoostModel class.

        Parameters:
        - params (dict): A dictionary containing the hyperparameters for the XGBoost model.
        """
        self.params = params
        self.model = None

    def train(self, dataloader: DataLoaderProtocol):
        """
        Trains the XGBoost model using the provided dataloader.

        Parameters:
        - dataloader (DataLoaderProtocol): An instance of DataLoaderProtocol that provides the training data.

        Returns:
        None
        """
        X_train, y_train = dataloader.get_train_data()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(self.params, dtrain)

    def predict(self, dataloader: DataLoaderProtocol):
        """
        Makes predictions using the trained XGBoost model on the test data from the dataloader.

        Parameters:
        - dataloader (DataLoaderProtocol): An instance of DataLoaderProtocol that provides the test data.

        Returns:
        - predictions (numpy.ndarray): An array of predicted labels for the test data.
        """
        X_test = dataloader.get_test_data()
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest)
