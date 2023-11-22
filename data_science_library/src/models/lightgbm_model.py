from data_science_library.src.protocals.dataloader import DataLoaderProtocol
from data_science_library.src.protocals.models import ModelsProtocol
import lightgbm as lgb

class LightGBMModel(ModelsProtocol):
    """
    A class representing a LightGBM model.

    This class implements the ModelsProtocol and provides methods for training and predicting using a LightGBM model.

    Attributes:
        params (dict): A dictionary containing the parameters for the LightGBM model.
        model: The trained LightGBM model.

    Methods:
        train(dataloader: DataLoaderProtocol): Trains the LightGBM model using the provided dataloader.
        predict(dataloader: DataLoaderProtocol): Makes predictions using the trained LightGBM model on the test data from the dataloader.
    """

    def __init__(self, params):
        """
        Initializes a LightGBMModel instance.

        Args:
            params (dict): A dictionary containing the parameters for the LightGBM model.
        """
        self.params = params
        self.model = None

    def train(self, dataloader: DataLoaderProtocol):
        """
        Trains the LightGBM model using the provided dataloader.

        Args:
            dataloader (DataLoaderProtocol): An instance of DataLoaderProtocol that provides the training data.

        Returns:
            None
        """
        X_train, y_train = dataloader.get_train_data()
        lgb_train = lgb.Dataset(X_train, y_train)
        self.model = lgb.train(self.params, lgb_train)

    def predict(self, dataloader: DataLoaderProtocol):
        """
        Makes predictions using the trained LightGBM model on the test data from the dataloader.

        Args:
            dataloader (DataLoaderProtocol): An instance of DataLoaderProtocol that provides the test data.

        Returns:
            numpy.ndarray: An array of predicted values.
        """
        X_test = dataloader.get_test_data()
        return self.model.predict(X_test)
