from abc import ABC, abstractmethod


class ModelsProtocol(ABC):
    """
    Abstract base class for models in the data science library.

    This class defines the interface for training and predicting using models.

    Attributes:
        None

    Methods:
        train(dataloader): Abstract method to train the model.
        predict(dataloader): Abstract method to make predictions using the model.
    """

    @abstractmethod
    def train(self, dataloader):
        """
        Train the model using the provided dataloader.

        Args:
            dataloader: The dataloader containing the training data.

        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, dataloader):
        """
        Make predictions using the model on the provided dataloader.

        Args:
            dataloader: The dataloader containing the data to make predictions on.

        Returns:
            None
        """
        pass
