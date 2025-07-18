Examples and Tutorials
=====================

This section provides detailed examples and tutorials for using the classification model utilities.

Complete Classification Pipeline
-----------------------------

Here's a complete example showing how to use all components together:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split
   from src.utils.preprocessing import DataPreprocessor
   from src.utils.feature_selection import FeatureSelector
   from src.utils.model_utils import ModelTrainer, ModelEvaluator
   from src.utils.model_factory import ModelFactory

   # Load data
   # Replace this with your data loading code
   df = pd.DataFrame({
       'age': [25, np.nan, 30, 45, 25],
       'income': [50000, 60000, np.nan, 75000, 200000],
       'category': ['A', 'B', np.nan, 'A', 'C'],
       'target': [0, 1, 1, 0, 1]
   })

   # Define features and target
   X = df.drop('target', axis=1)
   y = df['target']

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # Initialize preprocessing
   preprocessor = DataPreprocessor()

   # Define column types
   numeric_cols = ['age', 'income']
   categorical_cols = ['category']

   # Preprocess training data
   X_train_clean = preprocessor.handle_missing_values(X_train, numeric_cols, categorical_cols)
   X_train_clean = preprocessor.handle_outliers(X_train_clean, numeric_cols)
   X_train_clean = preprocessor.encode_categorical(X_train_clean, categorical_cols)
   X_train_clean = preprocessor.scale_features(X_train_clean, numeric_cols)

   # Preprocess test data using same transformations
   X_test_clean = preprocessor.handle_missing_values(X_test, numeric_cols, categorical_cols)
   X_test_clean = preprocessor.handle_outliers(X_test_clean, numeric_cols)
   X_test_clean = preprocessor.encode_categorical(X_test_clean, categorical_cols)
   X_test_clean = preprocessor.scale_features(X_test_clean, numeric_cols)

   # Feature selection
   selector = FeatureSelector()
   feature_importance, X_train_selected = selector.select_features_importance(
       X_train_clean, y_train, method='rf'
   )
   print("Feature Importance:\\n", feature_importance)

   # Select same features for test set
   X_test_selected = X_test_clean[selector.selected_features]

   # Create and train models
   models = {}
   for model_name in ['rf', 'svm', 'logistic']:
       # Get model and default parameters
       model = ModelFactory.get_model(model_name)
       param_grid = ModelFactory.get_default_params(model_name)
       
       # Train and optimize
       trainer = ModelTrainer(model)
       best_model, best_params = trainer.grid_search_cv(
           X_train_selected, y_train, param_grid
       )
       
       # Store model
       models[model_name] = best_model

   # Compare models
   comparison = ModelEvaluator.compare_models(models, X_test_selected, y_test)
   print("\\nModel Comparison:\\n", comparison)

   # Detailed evaluation of best model
   best_model_name = comparison['accuracy'].idxmax()
   best_model = models[best_model_name]
   
   y_pred = best_model.predict(X_test_selected)
   y_prob = best_model.predict_proba(X_test_selected)
   
   metrics, conf_matrix = ModelEvaluator.evaluate_classifier(
       y_test, y_pred, y_prob
   )
   
   print("\\nBest Model:", best_model_name)
   print("Metrics:", metrics)
   print("Confusion Matrix:\\n", conf_matrix)

Tips and Best Practices
---------------------

1. Data Preprocessing
   - Always handle missing values first
   - Apply outlier treatment only to numerical features
   - Use the same preprocessor instance for train and test data

2. Feature Selection
   - Try multiple feature selection methods
   - Consider domain knowledge when selecting features
   - Check for multicollinearity

3. Model Training
   - Start with default parameters
   - Use cross-validation to prevent overfitting
   - Try both grid search and Optuna for optimization

4. Model Evaluation
   - Look at multiple metrics, not just accuracy
   - Consider class imbalance
   - Use confusion matrix for detailed error analysis
