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
    #l = LightGBMModel(params=params)

    # predicting on training data
    #l.train(s)

    # predicting on test data
    #print(l.predict(s))
    
    #hp = LightGBMHyperparameterSearch(s.get_train_data(), (s.get_test_data(), s.y_test), params, task=settings.SAMPLE_LOADER_TYPE)
    #print(hp.optimize(study_name=settings.STUDY_NAME, n_trials=settings.N_TRIALS))

    
    params = {}
    if settings.SAMPLE_LOADER_TYPE == 'multiclass':
        params['objective'] = 'multi:softmax'
        params['num_class'] = 3  # Replace 3 with the appropriate number of classes
        params['eval_metric'] = 'mlogloss'
    elif settings.SAMPLE_LOADER_TYPE == 'binary':
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
    elif settings.SAMPLE_LOADER_TYPE == 'regression':
        params['objective'] = 'reg:squarederror'
        params['eval_metric'] = 'rmse'
        

    x  = XGBoostHyperparameterSearch(s.get_train_data(), (s.get_test_data(), s.y_test), params, task=settings.SAMPLE_LOADER_TYPE)
    print(x.optimize(study_name=settings.STUDY_NAME, n_trials=settings.N_TRIALS))