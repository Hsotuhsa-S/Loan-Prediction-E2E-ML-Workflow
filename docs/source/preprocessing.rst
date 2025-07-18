Data Preprocessing
=================

The ``DataPreprocessor`` class provides utilities for cleaning and preparing data for machine learning models.

.. automodule:: src.utils.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
------------

.. code-block:: python

   from src.utils.preprocessing import DataPreprocessor
   import pandas as pd
   import numpy as np

   # Create sample data
   df = pd.DataFrame({
       'age': [25, np.nan, 30, 45, 25],
       'income': [50000, 60000, np.nan, 75000, 200000],
       'category': ['A', 'B', np.nan, 'A', 'C']
   })

   # Initialize preprocessor
   preprocessor = DataPreprocessor()

   # Define column types
   numeric_cols = ['age', 'income']
   categorical_cols = ['category']

   # Handle missing values
   clean_df = preprocessor.handle_missing_values(df, numeric_cols, categorical_cols)

   # Handle outliers in numeric columns
   clean_df = preprocessor.handle_outliers(clean_df, numeric_cols, method='iqr')

   # Encode categorical variables
   clean_df = preprocessor.encode_categorical(clean_df, categorical_cols)

   # Scale numeric features
   clean_df = preprocessor.scale_features(clean_df, numeric_cols)

Key Features
-----------

1. Missing Value Treatment
   - Median imputation for numerical variables
   - Most frequent value imputation for categorical variables

2. Outlier Treatment
   - IQR method
   - Z-score method

3. Categorical Encoding
   - Label encoding with consistent mapping

4. Feature Scaling
   - Standardization using StandardScaler
