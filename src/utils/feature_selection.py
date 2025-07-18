import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class FeatureSelector:
    """A utility class for performing feature selection in machine learning pipelines.

    This class provides methods for selecting relevant features using different approaches:
    statistical tests (chi-square, ANOVA), model-based importance scores, and
    correlation analysis.

    Attributes:
        selected_features (list, optional): List of feature names that were selected
            by the last feature selection operation.
        feature_importances (pandas.DataFrame, optional): DataFrame containing feature
            importance scores from the last model-based feature selection.
    """

    def __init__(self):
        """Initialize the FeatureSelector with empty attributes."""
        self.selected_features = None
        self.feature_importances = None

    def select_features_statistical(self, X, y, method='chi2', k='all'):
        """Select features using statistical tests (chi-square or ANOVA F-value).

        Args:
            X (pandas.DataFrame): Feature matrix.
            y (array-like): Target variable.
            method (str, optional): Statistical test to use.
                'chi2': Chi-square test (features must be non-negative)
                'f_classif': ANOVA F-value test
                Defaults to 'chi2'.
            k (int or 'all', optional): Number of top features to select.
                If 'all', keeps all features with non-zero scores.
                Defaults to 'all'.

        Returns:
            tuple:
                - pandas.DataFrame: DataFrame with feature names and their scores,
                    sorted by score in descending order.
                - pandas.DataFrame: Selected features matrix.

        Note:
            For chi-square test, features are automatically shifted to be non-negative
            if negative values are present.

        Examples:
            >>> selector = FeatureSelector()
            >>> X = pd.DataFrame({
            ...     'feature1': [1, 2, 3],
            ...     'feature2': [4, 5, 6]
            ... })
            >>> y = [0, 1, 1]
            >>> scores, X_selected = selector.select_features_statistical(
            ...     X, y, method='f_classif', k=1
            ... )
        """
        if method == 'chi2':
            # Ensure all features are non-negative for chi-square test
            X = X - X.min() if (X.min() < 0).any() else X
            selector = SelectKBest(chi2, k=k)
        else:
            selector = SelectKBest(f_classif, k=k)
        
        selector.fit(X, y)
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        }).sort_values(by='Score', ascending=False)
        
        self.selected_features = X.columns[selector.get_support()].tolist()
        return feature_scores, X.iloc[:, selector.get_support()]

    def select_features_importance(self, X, y, method='rf', threshold=0.01):
        """Select features using model-based importance scores.

        Uses either Random Forest or Logistic Regression to compute feature
        importance scores and select features based on a threshold.

        Args:
            X (pandas.DataFrame): Feature matrix.
            y (array-like): Target variable.
            method (str, optional): Model to use for importance calculation.
                'rf': Random Forest (uses feature_importances_)
                'logistic': Logistic Regression (uses absolute coefficients)
                Defaults to 'rf'.
            threshold (float, optional): Minimum importance score for a feature
                to be selected. Defaults to 0.01.

        Returns:
            tuple:
                - pandas.DataFrame: DataFrame with feature names and their
                    importance scores, sorted by importance in descending order.
                - pandas.DataFrame: Selected features matrix containing only
                    features with importance > threshold.

        Note:
            - Random Forest uses the built-in feature_importances_ attribute
            - Logistic Regression uses the absolute values of coefficients
            - Results are stored in self.feature_importances and self.selected_features

        Examples:
            >>> selector = FeatureSelector()
            >>> X = pd.DataFrame({
            ...     'feature1': [1, 2, 3],
            ...     'feature2': [4, 5, 6]
            ... })
            >>> y = [0, 1, 1]
            >>> importances, X_selected = selector.select_features_importance(
            ...     X, y, method='rf', threshold=0.1
            ... )
        """
        if method == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(random_state=42)
        
        model.fit(X, y)
        
        if method == 'rf':
            importances = model.feature_importances_
        else:
            importances = np.abs(model.coef_[0])
        
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        self.feature_importances = feature_importances
        self.selected_features = feature_importances[
            feature_importances['Importance'] > threshold
        ]['Feature'].tolist()
        
        return feature_importances, X[self.selected_features]

    def correlation_filter(self, X, threshold=0.95):
        """Remove highly correlated features from the dataset.

        Computes the correlation matrix and removes one feature from each pair
        of features that have a correlation coefficient higher than the threshold.

        Args:
            X (pandas.DataFrame): Feature matrix.
            threshold (float, optional): Correlation coefficient threshold.
                Features with correlation higher than this value will be considered
                highly correlated. One feature from each highly correlated pair
                will be removed. Defaults to 0.95.

        Returns:
            pandas.DataFrame: DataFrame with highly correlated features removed.

        Note:
            - Uses absolute correlation coefficients
            - Only considers upper triangle of correlation matrix to avoid redundant
              comparisons
            - When a feature is highly correlated with multiple other features,
              it will be removed if it appears first in any pair above threshold

        Examples:
            >>> selector = FeatureSelector()
            >>> X = pd.DataFrame({
            ...     'feature1': [1, 2, 3],
            ...     'feature2': [2, 4, 6],  # Perfectly correlated with feature1
            ...     'feature3': [5, 7, 9]
            ... })
            >>> X_uncorrelated = selector.correlation_filter(X, threshold=0.9)
            >>> # feature2 will be removed due to high correlation with feature1
        """
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        return X.drop(to_drop, axis=1)
