# 🚀 Super Awesome DS Project 🚀

Welcome to the Super Awesome DS Project! This project is the result of countless hours of coding, gallons of coffee, and an insatiable passion for DS. 

## 🎯 What's this all about?

This project is all about making ds easier. It's designed to quickly build a model.  It has protocols to extend it futher. .

## Installation

TBD


## 🏁 Getting Started

This project uses the `Dynaconf` library for configuration management. The configuration parameters are stored in a `settings.toml` file.

## Configuration

The `settings.toml` file contains sections for different environments (`default`, `preprod`, `prod`). Each section contains the following parameters:

- `ENVIRONMENT`: The name of the environment.
- `SAMPLE_LOADER_TYPE`: The type of sample loader to use.
- `LIGHTGBM_PARAMS`: The parameters for the LightGBM model.
- `STUDY_NAME`: The name of the study for the hyperparameter search.
- `N_TRIALS`: The number of trials for the hyperparameter search.

## Code

Here is the main script of the project:

```python
from data_science_library.src.dataloader.sampleloader import SampleLoader
from data_science_library.src.models.lightgbm_model import LightGBMModel
from data_science_library.src.hyperparameter.xgboost_hyperparameter_search import XGBoostHyperparameterSearch
from data_science_library.src.hyperparameter.lightgbm_hyperparameter_search import LightGBMHyperparameterSearch
from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="DYNACONF",  # environment variables prefix
    environments=True,  # enable the use of [default], [development], [production] in TOML files
    settings_files=['settings.toml'],  # point to the settings file
)

def main():
    """
    Demonstration of how to use the data science library.

    This function showcases the usage of the data science library by performing the following steps:
    1. Loads a sample dataset for regression.
    2. Prepares the data for training and testing.
    3. Instantiates a LightGBM model.
    4. Trains the model on the training data.
    5. Predicts the target variable for the test data using the trained model.
    6. Performs hyperparameter search using LightGBM and XGBoost models.

    Returns:
        None
    """
    
    print(settings.ENVIRONMENT)
    print(settings.SAMPLE_LOADER_TYPE)
    print(settings.STUDY_NAME)
    print(settings.N_TRIALS)
    
    s = SampleLoader(settings.SAMPLE_LOADER_TYPE)
    _ = s.prepare_data() 

    # instantiating the parameters for the model
    params = {'objective': settings.SAMPLE_LOADER_TYPE}
    if settings.SAMPLE_LOADER_TYPE == 'multiclass':
        params['num_class'] = 3  # Replace 3 with the appropriate number of classes

     # instantiating the parameters for the model
    l = LightGBMModel(params=params)

    # predicting on training data
    l.train(s)

    # predicting on test data
    print(l.predict(s))
    
    hp = LightGBMHyperparameterSearch(s.get_train_data(), (s.get_test_data(), s.y_test), params, task=settings.SAMPLE_LOADER_TYPE)
    print(hp.optimize(study_name=settings.STUDY_NAME, n_trials=settings.N_TRIALS))

    #x  = XGBoostHyperparameterSearch(s.get_train_data(), (s.get_test_data(), s.y_test), {}, task=settings.SAMPLE_LOADER_TYPE)
    #print(x.optimize(study_name=settings.STUDY_NAME, n_trials=settings.N_TRIALS))

## 📚 Documentation

TBD

## 🎉 Contributing

TBD

## 📃 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## 💖 Thanks

Thanks for checking out our project! We hope you enjoy using it as much as we enjoyed building it. Happy coding! 🎉