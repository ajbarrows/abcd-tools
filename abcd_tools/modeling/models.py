"""Machine learning models and results collection.

This module provides functions for:
- ElasticNet cross-validation model training using glmnetpy
- Experiment execution and results collection
- Results storage and retrieval
"""

from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from glmnet import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from abcd_tools.modeling.preprocessing import prepare_for_preprocessing


def enet_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_inner_folds: int = 5,
    n_alphas: int = 100,
    l1_ratio: float = 0.5,
    random_state: int = 42,
) -> Tuple[Dict, List[float]]:
    """Train ElasticNet models using nested cross-validation with glmnetpy.

    Uses outer K-fold CV for evaluation and inner CV for hyperparameter tuning.
    glmnetpy uses the fast Fortran implementation from the R glmnet package.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    y : np.ndarray
        Target vector, shape (n_samples,)
    n_splits : int, optional
        Number of outer CV folds (default: 5)
    n_inner_folds : int, optional
        Number of inner CV folds for hyperparameter tuning (default: 5)
    n_alphas : int, optional
        Number of lambda values to test (default: 100)
    l1_ratio : float, optional
        ElasticNet mixing parameter, 0 <= l1_ratio <= 1 (default: 0.5)
        - l1_ratio = 1 is Lasso
        - l1_ratio = 0 is Ridge
    random_state : int, optional
        Random seed for reproducibility (default: 42)

    Returns
    -------
    models : dict
        Dictionary mapping fold names to {'model': ElasticNet, 'score': float}
    cv_scores : list of float
        Test set R² scores for each outer fold

    Notes
    -----
    glmnetpy parameter mapping:
    - alpha (glmnetpy) = l1_ratio (sklearn): L1/L2 mixing parameter
    - n_lambda (glmnetpy) = n_alphas (sklearn): number of regularization values
    - lambda (glmnetpy internal) = alpha (sklearn): regularization strength
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = {}
    cv_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model with inner CV for hyperparameter tuning
        # glmnetpy's ElasticNet has built-in CV
        model = ElasticNet(
            alpha=l1_ratio,  # L1/L2 mixing (1=Lasso, 0=Ridge)
            n_splits=n_inner_folds,  # Inner CV folds
            scoring="r2",  # Use R² for model selection
            random_state=random_state,
            n_lambda=n_alphas,  # Number of lambda values to try
        )
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        cv_scores.append(score)

        models[f"cv_fold_{fold}"] = {"model": model, "score": score}

    return models, cv_scores


def run_single_experiment(
    task: str,
    condition: str,
    experiment: Union[Tuple[str, str, str], Tuple[str, str, str, str]],
    outcome: str,
    data: pd.DataFrame,
    phenotype: pd.DataFrame,
    params: Dict,
) -> Dict:
    """Run a single experiment configuration.

    Parameters
    ----------
    task : str
        Task name
    condition : str
        Condition name
    experiment : tuple of (str, str, str) or (str, str, str, str)
        Experiment configuration: (normalize, outliers, parcellation) or
        (normalize, outliers, parcellation, parcellation_timing)
    outcome : str
        Outcome variable to predict
    data : pd.DataFrame
        Prepared data for this experiment
    phenotype : pd.DataFrame
        Phenotype data
    params : dict
        Configuration parameters including n_splits, n_inner_folds, etc.

    Returns
    -------
    dict
        Result dictionary with task, condition, experiment, outcome, scores, models,
        n_features, and n_samples
    """
    X, y = prepare_for_preprocessing(data, phenotype, outcome)

    models, scores = enet_cv(
        X,
        y,
        n_splits=params["n_splits"],
        n_inner_folds=params["n_inner_folds"],
        n_alphas=params["n_alphas"],
        l1_ratio=params["l1_ratio"],
        random_state=params["random_state"],
    )

    return {
        "task": task,
        "condition": condition,
        "experiment": experiment,
        "outcome": outcome,
        "scores": scores,
        "models": models,
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
    }


class ExperimentResults:
    """Simple container for collecting and storing experiment results.

    Stores summary statistics as records that can be converted to a DataFrame,
    and optionally stores detailed model objects.

    Parameters
    ----------
    save_path : str, optional
        Default directory for saving results. Used by save_incremental()
        if no path is provided in the method call.

    Examples
    --------
    Batch saving (all results at once):
    >>> results = ExperimentResults()
    >>> results.add_result(
    ...     task='nback', condition='0b', experiment=('none', 'none', 'destrieux', 'after'),
    ...     outcome='nback_0b_rt', scores=[0.1, 0.2, 0.15, 0.18, 0.12],
    ...     models=models_dict, n_features=148, n_samples=100
    ... )
    >>> results.save('./data/results')
    >>> df = results.to_dataframe()

    Incremental saving (each result saved as it's added):
    >>> results = ExperimentResults(save_path='./data/results')
    >>> results.save_incremental(
    ...     task='nback', condition='0b', experiment=('none', 'none', 'destrieux', 'after'),
    ...     outcome='nback_0b_rt', scores=[0.1, 0.2, 0.15], n_features=148, n_samples=100
    ... )
    """

    def __init__(self, save_path: Optional[str] = None):
        self.summary_records = []
        self.save_path = Path(save_path) if save_path else None

    def add_result(
        self,
        task: str,
        condition: str,
        experiment: Union[Tuple[str, str, str], Tuple[str, str, str, str]],
        outcome: str,
        scores: List[float],
        n_features: int,
        n_samples: int,
        models: Optional[Dict] = None,
    ):
        """Add a single experiment result.

        Parameters
        ----------
        task : str
            Task name (e.g., 'nback', 'sst')
        condition : str
            Condition name (e.g., '0b', '2b')
        experiment : tuple of (str, str, str) or (str, str, str, str)
            Experiment configuration: (normalize, outliers, parcellation) or
            (normalize, outliers, parcellation, parcellation_timing)
        outcome : str
            Phenotype being predicted
        scores : list of float
            Cross-validation R² scores from each fold
        n_features : int
            Number of features in the model
        n_samples : int
            Number of samples used
        models : dict, optional
            Dictionary of trained models (not stored in summary)
        """
        # Handle both 3-tuple and 4-tuple experiments
        if len(experiment) == 4:
            normalize, outliers, parcellation, parcellation_timing = experiment
        else:
            normalize, outliers, parcellation = experiment
            parcellation_timing = None  # Not specified for older experiments

        record = {
            "task": task,
            "condition": condition,
            "outcome": outcome,
            "normalize": normalize,
            "outliers": outliers,
            "parcellation": parcellation,
            "parcellation_timing": parcellation_timing,
            "n_samples": n_samples,
            "n_features": n_features,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
        }

        # Add individual fold scores
        for i, score in enumerate(scores):
            record[f"fold_{i}_score"] = score

        self.summary_records.append(record)

    def save_incremental(
        self,
        task: str,
        condition: str,
        experiment: Union[Tuple[str, str, str], Tuple[str, str, str, str]],
        outcome: str,
        scores: List[float],
        n_features: int,
        n_samples: int,
        models: Optional[Dict] = None,
        save_path: Optional[str] = None,
    ):
        """Add a result and immediately save it to disk incrementally.

        This method allows saving each experiment result as it completes,
        rather than waiting until all experiments finish. Useful for long-running
        experiments where you want to preserve progress.

        Parameters
        ----------
        task : str
            Task name (e.g., 'nback', 'sst')
        condition : str
            Condition name (e.g., '0b', '2b')
        experiment : tuple of (str, str, str) or (str, str, str, str)
            Experiment configuration: (normalize, outliers, parcellation) or
            (normalize, outliers, parcellation, parcellation_timing)
        outcome : str
            Phenotype being predicted
        scores : list of float
            Cross-validation R² scores from each fold
        n_features : int
            Number of features in the model
        n_samples : int
            Number of samples used
        models : dict, optional
            Dictionary of trained models (saved separately if provided)
        save_path : str, optional
            Directory to save results. If None, uses self.save_path set in __init__

        Examples
        --------
        >>> results = ExperimentResults(save_path='./data/results')
        >>> # After each experiment completes:
        >>> results.save_incremental(
        ...     task='nback', condition='0b', experiment=('none', 'none', 'destrieux', 'after'),
        ...     outcome='nback_0b_rt', scores=[0.1, 0.2, 0.15], n_features=148, n_samples=100
        ... )
        """
        # Add the result to memory
        self.add_result(
            task=task,
            condition=condition,
            experiment=experiment,
            outcome=outcome,
            scores=scores,
            n_features=n_features,
            n_samples=n_samples,
            models=models,
        )

        # Determine save path
        path = Path(save_path) if save_path else self.save_path
        if path is None:
            raise ValueError(
                "No save path specified. Either set save_path in __init__ "
                "or pass it to save_incremental()"
            )

        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)

        # Get the record we just added
        record = self.summary_records[-1]

        # Convert to DataFrame row
        df_row = pd.DataFrame([record])

        # Append to parquet file
        parquet_path = path / "experiment_summary.parquet"
        if parquet_path.exists():
            # Load existing data and append
            existing_df = pd.read_parquet(parquet_path)
            combined_df = pd.concat([existing_df, df_row], ignore_index=True)
            combined_df.to_parquet(parquet_path, index=False)
        else:
            # Create new file
            df_row.to_parquet(parquet_path, index=False)

        # Also update CSV file
        csv_path = path / "experiment_summary.csv"
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, df_row], ignore_index=True)
            combined_df.to_csv(csv_path, index=False)
        else:
            df_row.to_csv(csv_path, index=False)

        # Optionally save models separately
        if models is not None:
            # Handle both 3-tuple and 4-tuple experiments
            if len(experiment) == 4:
                normalize, outliers, parcellation, parcellation_timing = experiment
                model_filename = f"{task}_{condition}_{outcome}_{normalize}_{outliers}_{parcellation}_{parcellation_timing}_models.pkl"
            else:
                normalize, outliers, parcellation = experiment
                model_filename = f"{task}_{condition}_{outcome}_{normalize}_{outliers}_{parcellation}_models.pkl"

            model_path = path / "models" / model_filename
            model_path.parent.mkdir(parents=True, exist_ok=True)

            with open(model_path, "wb") as f:
                pickle.dump(models, f)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert summary records to DataFrame.

        Returns
        -------
        pd.DataFrame
            Summary statistics with multi-index (task, condition, outcome)
        """
        df = pd.DataFrame(self.summary_records)

        if len(df) > 0:
            df = df.set_index(["task", "condition", "outcome"])

        return df

    def save(self, base_path: str):
        """Save results to disk as parquet and CSV.

        Parameters
        ----------
        base_path : str
            Directory to save results
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        df.to_csv(base_path / "experiment_summary.csv")
        df.to_parquet(base_path / "experiment_summary.parquet")

    @classmethod
    def load(cls, base_path: str) -> "ExperimentResults":
        """Load results from disk.

        Parameters
        ----------
        base_path : str
            Directory containing saved results

        Returns
        -------
        ExperimentResults
            Loaded results object
        """
        base_path = Path(base_path)

        results = cls()
        df = pd.read_parquet(base_path / "experiment_summary.parquet")
        results.summary_records = df.reset_index().to_dict("records")

        return results
