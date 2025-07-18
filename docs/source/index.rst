Classification Models Python Documentation
=====================================

Welcome to the Classification Models Python documentation. This project provides a comprehensive set of utilities for building and evaluating machine learning classification models.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   preprocessing
   feature_selection
   model_utils
   examples

Getting Started
--------------

Installation
~~~~~~~~~~~
To use these utilities, first install the required packages:

.. code-block:: bash

   pip install numpy pandas scikit-learn xgboost lightgbm optuna

Quick Start
~~~~~~~~~~

Here's a simple example of using the utilities:

.. code-block:: python

   from src.utils.preprocessing import DataPreprocessor
   from src.utils.feature_selection import FeatureSelector
   from src.utils.model_utils import ModelTrainer, ModelEvaluator
   from src.utils.model_factory import ModelFactory

   # Initialize preprocessing
   preprocessor = DataPreprocessor()

   # Clean and preprocess data
   clean_data = preprocessor.handle_missing_values(df, numeric_cols, categorical_cols)
   clean_data = preprocessor.handle_outliers(clean_data, numeric_cols)
   clean_data = preprocessor.encode_categorical(clean_data, categorical_cols)
   clean_data = preprocessor.scale_features(clean_data, numeric_cols)

   # Select features
   selector = FeatureSelector()
   feature_importance, X_selected = selector.select_features_importance(X, y)

   # Create and train model
   model = ModelFactory.get_model('rf')  # Get Random Forest model
   trainer = ModelTrainer(model)
   
   # Optimize hyperparameters
   param_grid = ModelFactory.get_default_params('rf')
   best_model, best_params = trainer.grid_search_cv(X_train, y_train, param_grid)

   # Evaluate model
   metrics, conf_matrix = ModelEvaluator.evaluate_classifier(y_test, y_pred, y_prob)

Components
---------

The project consists of several main components:

1. :doc:`preprocessing`
   - Missing value imputation
   - Outlier treatment
   - Categorical encoding
   - Feature scaling

2. :doc:`feature_selection`
   - Statistical feature selection
   - Model-based feature importance
   - Correlation filtering

3. :doc:`model_utils`
   - Model training
   - Hyperparameter optimization
   - Model evaluation and comparison

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
