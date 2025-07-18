Feature Selection
=================

The ``FeatureSelector`` class provides methods for selecting the most relevant features for your machine learning models.

.. automodule:: src.utils.feature_selection
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
------------

.. code-block:: python

   from src.utils.feature_selection import FeatureSelector
   import pandas as pd
   import numpy as np

   # Create sample data
   X = pd.DataFrame({
       'feature1': [1, 2, 3, 4, 5],
       'feature2': [2, 4, 6, 8, 10],  # Perfectly correlated with feature1
       'feature3': [1, 3, 2, 4, 5]
   })
   y = [0, 0, 1, 1, 1]

   # Initialize selector
   selector = FeatureSelector()

   # Statistical feature selection
   scores, X_selected = selector.select_features_statistical(X, y, method='f_classif')
   print("Feature scores:\\n", scores)

   # Model-based feature importance
   importances, X_selected = selector.select_features_importance(X, y, method='rf')
   print("Feature importances:\\n", importances)

   # Remove highly correlated features
   X_uncorrelated = selector.correlation_filter(X, threshold=0.9)

Key Features
-----------

1. Statistical Feature Selection
   - Chi-square test for categorical features
   - ANOVA F-value test for numerical features

2. Model-based Feature Importance
   - Random Forest importance scores
   - Logistic Regression coefficients

3. Correlation Filtering
   - Remove highly correlated features
   - Customizable correlation threshold
