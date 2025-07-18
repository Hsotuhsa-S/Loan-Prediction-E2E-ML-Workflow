import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)
import optuna

class ModelTrainer:
    """A class for training and optimizing machine learning models.

    This class provides methods for training models with different optimization
    strategies including basic training, grid search cross-validation, and
    Optuna-based hyperparameter optimization.

    Attributes:
        model: The base model to be trained (any sklearn-compatible model).
        best_model: The best model found during optimization (if performed).
        best_params (dict): The best parameters found during optimization.
    """

    def __init__(self, model):
        """Initialize the ModelTrainer with a base model.

        Args:
            model: Any sklearn-compatible model instance (e.g., RandomForestClassifier).
        """
        self.model = model
        self.best_model = None
        self.best_params = None
        
    def train(self, X_train, y_train):
        """Train the model on the given data.

        Performs basic model training without any optimization.

        Args:
            X_train (array-like): Training data features.
            y_train (array-like): Training data labels.

        Returns:
            object: The trained model instance.

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> trainer = ModelTrainer(RandomForestClassifier())
            >>> X_train = [[1, 2], [3, 4]]
            >>> y_train = [0, 1]
            >>> model = trainer.train(X_train, y_train)
        """
        self.model.fit(X_train, y_train)
        return self.model
    
    def grid_search_cv(self, X_train, y_train, param_grid, cv=5, scoring='accuracy'):
        """Perform grid search cross-validation for hyperparameter optimization.

        Args:
            X_train (array-like): Training data features.
            y_train (array-like): Training data labels.
            param_grid (dict): Dictionary with parameters names (string) as keys and
                lists of parameter settings to try as values.
            cv (int, optional): Number of folds for cross-validation.
                Defaults to 5.
            scoring (str, optional): Scoring metric to use for evaluation.
                Defaults to 'accuracy'.

        Returns:
            tuple:
                - object: The best model found during grid search.
                - dict: The parameters of the best model.

        Note:
            The best model and parameters are also stored in self.best_model
            and self.best_params respectively.

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> trainer = ModelTrainer(RandomForestClassifier())
            >>> param_grid = {
            ...     'n_estimators': [100, 200],
            ...     'max_depth': [10, 20]
            ... }
            >>> best_model, best_params = trainer.grid_search_cv(
            ...     X_train, y_train, param_grid
            ... )
        """
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def optuna_optimize(self, X_train, y_train, param_space, n_trials=100):
        """Perform hyperparameter optimization using Optuna.

        Uses Optuna for efficient hyperparameter optimization with smart
        sampling strategies.

        Args:
            X_train (array-like): Training data features.
            y_train (array-like): Training data labels.
            param_space (dict): Dictionary defining the search space for each parameter.
                Format: {param_name: (min_value, max_value)} for numerical parameters
                or {param_name: [possible_values]} for categorical parameters.
            n_trials (int, optional): Number of optimization trials.
                Defaults to 100.

        Returns:
            tuple:
                - object: The best model found during optimization.
                - dict: The best parameters found.

        Note:
            - Automatically determines parameter types and uses appropriate Optuna
              suggestion methods.
            - Uses 5-fold cross-validation for model evaluation.
            - The best model and parameters are stored in self.best_model
              and self.best_params.

        Examples:
            >>> trainer = ModelTrainer(RandomForestClassifier())
            >>> param_space = {
            ...     'n_estimators': (100, 300),
            ...     'max_depth': (10, 30),
            ...     'criterion': ['gini', 'entropy']
            ... }
            >>> best_model, best_params = trainer.optuna_optimize(
            ...     X_train, y_train, param_space
            ... )
        """
        def objective(trial):
            params = {
                key: (trial.suggest_int if isinstance(val[0], int) else 
                     trial.suggest_float if isinstance(val[0], float) else
                     trial.suggest_categorical)(key, *val)
                for key, val in param_space.items()
            }
            
            self.model.set_params(**params)
            score = cross_val_score(
                self.model, X_train, y_train, cv=5, scoring='accuracy'
            ).mean()
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        self.model.set_params(**self.best_params)
        self.best_model = self.model
        
        return self.best_model, self.best_params

class ModelEvaluator:
    """A utility class for evaluating and comparing classification models.

    This class provides static methods for computing various classification
    metrics and comparing multiple models' performance.
    """

    @staticmethod
    def evaluate_classifier(y_true, y_pred, y_prob=None):
        """Evaluate a classifier using multiple performance metrics.

        Computes various classification metrics including accuracy, precision,
        recall, F1-score, ROC AUC (if probabilities are provided), and
        confusion matrix.

        Args:
            y_true (array-like): Ground truth (correct) labels.
            y_pred (array-like): Predicted labels.
            y_prob (array-like, optional): Predicted probabilities.
                For binary classification: array of shape (n_samples, 2)
                For multiclass: array of shape (n_samples, n_classes)
                Defaults to None.

        Returns:
            tuple:
                - dict: Dictionary containing various metrics:
                    - 'accuracy': Overall accuracy score
                    - 'precision': Precision score (weighted average)
                    - 'recall': Recall score (weighted average)
                    - 'f1': F1 score (weighted average)
                    - 'roc_auc': ROC AUC score (if y_prob is provided)
                - array: Confusion matrix

        Examples:
            >>> evaluator = ModelEvaluator()
            >>> y_true = [0, 1, 1, 0]
            >>> y_pred = [0, 1, 0, 0]
            >>> y_prob = [[0.8, 0.2], [0.3, 0.7],
            ...          [0.6, 0.4], [0.9, 0.1]]
            >>> metrics, conf_matrix = evaluator.evaluate_classifier(
            ...     y_true, y_pred, y_prob
            ... )
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
        }
        
        if y_prob is not None:
            if y_prob.shape[1] == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:  # Multi-class
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='weighted'
                )
        
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        return metrics, conf_matrix
    
    @staticmethod
    def compare_models(models, X_test, y_test):
        """Compare multiple models using various performance metrics.

        Evaluates and compares multiple models on the same test data using
        the evaluate_classifier method.

        Args:
            models (dict): Dictionary of models to compare.
                Format: {'model_name': model_instance}
            X_test (array-like): Test data features.
            y_test (array-like): Test data labels.

        Returns:
            pandas.DataFrame: DataFrame containing performance metrics for each model.
                Each row represents a model, and columns are different metrics
                (accuracy, precision, recall, f1, roc_auc if applicable).

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.svm import SVC
            >>> evaluator = ModelEvaluator()
            >>> models = {
            ...     'Random Forest': RandomForestClassifier(),
            ...     'SVM': SVC(probability=True)
            ... }
            >>> # Assume models are already trained
            >>> comparison = evaluator.compare_models(
            ...     models, X_test, y_test
            ... )
            >>> print(comparison)  # Shows metrics for each model
        """
        results = []
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            metrics, _ = ModelEvaluator.evaluate_classifier(y_test, y_pred, y_prob)
            metrics['model'] = name
            results.append(metrics)
        
        return pd.DataFrame(results).set_index('model')
