Model Utilities
==============

The model utilities module provides classes for training, optimizing, and evaluating machine learning models.

Model Trainer
------------

.. autoclass:: src.utils.model_utils.ModelTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Model Evaluator
-------------

.. autoclass:: src.utils.model_utils.ModelEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
------------

Training and Optimizing Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.utils.model_utils import ModelTrainer
   from sklearn.ensemble import RandomForestClassifier

   # Initialize trainer with a model
   model = RandomForestClassifier()
   trainer = ModelTrainer(model)

   # Basic training
   trained_model = trainer.train(X_train, y_train)

   # Grid search optimization
   param_grid = {
       'n_estimators': [100, 200],
       'max_depth': [10, 20],
       'min_samples_split': [2, 5]
   }
   best_model, best_params = trainer.grid_search_cv(X_train, y_train, param_grid)

   # Optuna optimization
   param_space = {
       'n_estimators': (100, 300),
       'max_depth': (10, 30),
       'min_samples_split': (2, 10)
   }
   best_model, best_params = trainer.optuna_optimize(X_train, y_train, param_space)

Evaluating Models
~~~~~~~~~~~~~~

.. code-block:: python

   from src.utils.model_utils import ModelEvaluator
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.svm import SVC

   # Create and train multiple models
   models = {
       'Random Forest': RandomForestClassifier(),
       'SVM': SVC(probability=True)
   }
   
   # Train models...

   # Compare models
   comparison = ModelEvaluator.compare_models(models, X_test, y_test)
   print("Model Comparison:\\n", comparison)

   # Detailed evaluation of one model
   y_pred = best_model.predict(X_test)
   y_prob = best_model.predict_proba(X_test)
   metrics, conf_matrix = ModelEvaluator.evaluate_classifier(y_test, y_pred, y_prob)
   print("Metrics:", metrics)
   print("Confusion Matrix:\\n", conf_matrix)

Key Features
-----------

1. Model Training
   - Basic model training
   - Grid search cross-validation
   - Optuna-based hyperparameter optimization

2. Model Evaluation
   - Multiple classification metrics
   - ROC AUC score calculation
   - Confusion matrix generation
   - Model comparison utilities
