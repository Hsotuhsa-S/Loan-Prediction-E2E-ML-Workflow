from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class ModelFactory:
    """A factory class for creating and configuring different classification models.

    This class provides static methods to create various classification models and
    retrieve their default hyperparameter search spaces. Supported models include:
    Logistic Regression, SVM, Random Forest, KNN, Decision Tree, XGBoost, and LightGBM.

    Attributes:
        None
    """

    @staticmethod
    def get_model(model_name, **kwargs):
        """Creates and returns a classification model instance.

        Args:
            model_name (str): Name of the model to create. Supported values are:
                'logistic': Logistic Regression
                'svm': Support Vector Machine
                'rf': Random Forest
                'knn': K-Nearest Neighbors
                'dt': Decision Tree
                'xgb': XGBoost
                'lgbm': LightGBM
            **kwargs: Arbitrary keyword arguments passed to the model constructor.

        Returns:
            object: An instance of the requested model with specified parameters.

        Raises:
            ValueError: If the specified model_name is not supported.

        Examples:
            >>> factory = ModelFactory()
            >>> # Create a Random Forest with 100 estimators
            >>> rf_model = factory.get_model('rf', n_estimators=100)
            >>> # Create an SVM with RBF kernel
            >>> svm_model = factory.get_model('svm', kernel='rbf')
        """
        models = {
            'logistic': LogisticRegression,
            'svm': SVC,
            'rf': RandomForestClassifier,
            'knn': KNeighborsClassifier,
            'dt': DecisionTreeClassifier,
            'xgb': XGBClassifier,
            'lgbm': LGBMClassifier
        }
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not supported")
        
        return models[model_name](**kwargs)
    
    @staticmethod
    def get_default_params(model_name):
        """Returns the default hyperparameter search space for the specified model.

        This method provides a curated set of hyperparameters for each model type
        that can be used for model tuning via grid search or random search.

        Args:
            model_name (str): Name of the model to get parameters for. Supported values are:
                'logistic': Logistic Regression parameters (C, penalty, solver)
                'svm': SVM parameters (C, kernel, gamma)
                'rf': Random Forest parameters (n_estimators, max_depth, min_samples_split, min_samples_leaf)
                'knn': KNN parameters (n_neighbors, weights, metric)
                'dt': Decision Tree parameters (max_depth, min_samples_split, min_samples_leaf)
                'xgb': XGBoost parameters (n_estimators, max_depth, learning_rate, subsample)
                'lgbm': LightGBM parameters (n_estimators, max_depth, learning_rate, num_leaves)

        Returns:
            dict: A dictionary containing parameter names and their possible values.
                  The structure is: {'param_name': [possible_values]}

        Raises:
            ValueError: If the specified model_name is not supported.

        Examples:
            >>> factory = ModelFactory()
            >>> # Get Random Forest parameters
            >>> rf_params = factory.get_default_params('rf')
            >>> # Use parameters in grid search
            >>> grid_search = GridSearchCV(RandomForestClassifier(), rf_params)
        """
        param_spaces = {
            'logistic': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'dt': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgb': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0]
            },
            'lgbm': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [31, 62, 127]
            }
        }
        
        if model_name not in param_spaces:
            raise ValueError(f"Model {model_name} not supported")
        
        return param_spaces[model_name]
