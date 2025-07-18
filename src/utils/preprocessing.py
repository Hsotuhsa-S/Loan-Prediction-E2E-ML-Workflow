import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """A utility class for preprocessing data in machine learning pipelines.

    This class provides methods for common data preprocessing tasks including:
    handling missing values, treating outliers, encoding categorical variables,
    and scaling numerical features.

    Attributes:
        num_imputer (SimpleImputer): Imputer for handling missing values in numerical columns,
            uses median strategy by default.
        cat_imputer (SimpleImputer): Imputer for handling missing values in categorical columns,
            uses most frequent strategy by default.
        scaler (StandardScaler): Scaler for standardizing numerical features.
        label_encoders (dict): Dictionary to store LabelEncoder objects for each categorical column.
    """

    def __init__(self):
        """Initialize the DataPreprocessor with default preprocessing objects."""
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def handle_missing_values(self, df, numeric_cols, categorical_cols):
        """Handle missing values in both numeric and categorical columns.

        Applies median imputation for numerical columns and most frequent value
        imputation for categorical columns.

        Args:
            df (pandas.DataFrame): Input dataframe containing missing values.
            numeric_cols (list): List of column names containing numerical data.
            categorical_cols (list): List of column names containing categorical data.

        Returns:
            pandas.DataFrame: A copy of the input dataframe with missing values imputed.

        Examples:
            >>> preprocessor = DataPreprocessor()
            >>> df = pd.DataFrame({
            ...     'age': [25, np.nan, 30],
            ...     'category': ['A', np.nan, 'B']
            ... })
            >>> clean_df = preprocessor.handle_missing_values(
            ...     df, 
            ...     numeric_cols=['age'],
            ...     categorical_cols=['category']
            ... )
        """
        df_copy = df.copy()
        
        if numeric_cols:
            df_copy[numeric_cols] = self.num_imputer.fit_transform(df_copy[numeric_cols])
        
        if categorical_cols:
            df_copy[categorical_cols] = self.cat_imputer.fit_transform(df_copy[categorical_cols])
        
        return df_copy

    def handle_outliers(self, df, columns, method='iqr', threshold=1.5):
        """Handle outliers in numerical columns using either IQR or Z-score method.

        Args:
            df (pandas.DataFrame): Input dataframe containing outliers.
            columns (list): List of column names to check for outliers.
            method (str, optional): Method to use for outlier detection.
                'iqr': Interquartile Range method
                'zscore': Z-score method
                Defaults to 'iqr'.
            threshold (float, optional): Threshold for outlier detection.
                For IQR method: Points beyond Q1 - threshold*IQR and Q3 + threshold*IQR are outliers.
                For Z-score method: Points beyond threshold standard deviations are outliers.
                Defaults to 1.5.

        Returns:
            pandas.DataFrame: A copy of the input dataframe with outliers handled.
                For IQR method: Outliers are clipped to the bounds.
                For Z-score method: Outliers are replaced with the mean.

        Examples:
            >>> preprocessor = DataPreprocessor()
            >>> df = pd.DataFrame({
            ...     'values': [1, 2, 3, 100, 4, 5]
            ... })
            >>> # Using IQR method
            >>> clean_df = preprocessor.handle_outliers(
            ...     df, 
            ...     columns=['values'],
            ...     method='iqr',
            ...     threshold=1.5
            ... )
        """
        df_copy = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                df_copy[col] = df_copy[col].clip(lower_bound, upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                df_copy[col] = df_copy[col].mask(z_scores > threshold, df_copy[col].mean())
        
        return df_copy

    def encode_categorical(self, df, categorical_cols):
        """Encode categorical variables using Label Encoding.

        Converts categorical variables into numerical format using LabelEncoder.
        Maintains consistency in encoding across multiple calls by storing
        the LabelEncoder objects.

        Args:
            df (pandas.DataFrame): Input dataframe containing categorical variables.
            categorical_cols (list): List of column names containing categorical data.

        Returns:
            pandas.DataFrame: A copy of the input dataframe with categorical variables encoded.

        Note:
            The method stores LabelEncoder objects for each column to ensure
            consistent encoding across multiple transformations (e.g., train and test sets).

        Examples:
            >>> preprocessor = DataPreprocessor()
            >>> df = pd.DataFrame({
            ...     'category': ['A', 'B', 'A', 'C']
            ... })
            >>> encoded_df = preprocessor.encode_categorical(df, ['category'])
            >>> # The same encoding will be used for new data
            >>> new_df = pd.DataFrame({'category': ['B', 'C']})
            >>> encoded_new_df = preprocessor.encode_categorical(new_df, ['category'])
        """
        df_copy = df.copy()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_copy[col] = self.label_encoders[col].fit_transform(df_copy[col].astype(str))
        
        return df_copy

    def scale_features(self, df, columns):
        """Scale numerical features using StandardScaler.

        Standardizes features by removing the mean and scaling to unit variance
        using sklearn's StandardScaler.

        Args:
            df (pandas.DataFrame): Input dataframe containing numerical features.
            columns (list): List of column names to be scaled.

        Returns:
            pandas.DataFrame: A copy of the input dataframe with specified columns scaled.

        Note:
            The scaler is fit on the provided data and can be reused for
            transforming new data (e.g., test set) to ensure consistent scaling.

        Examples:
            >>> preprocessor = DataPreprocessor()
            >>> df = pd.DataFrame({
            ...     'feature1': [1, 2, 3],
            ...     'feature2': [10, 20, 30]
            ... })
            >>> scaled_df = preprocessor.scale_features(
            ...     df, 
            ...     columns=['feature1', 'feature2']
            ... )
            >>> # Features will have mean=0 and std=1
        """
        df_copy = df.copy()
        df_copy[columns] = self.scaler.fit_transform(df_copy[columns])
        return df_copy
