from __future__ import annotations
from typing import Any, Sequence, Optional, Dict, Union, List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from ..dataset import Dataset, TargetRole
from ..dataset.backends import PandasDataset
from .abstract import MLExtension
from ..utils.models import CUPAC_MODELS
import warnings


def create_multilevel_cupac_features(
    features_mapping: Dict[str, Dict[Tuple[str, int], List[str]]]
) -> Dict[str, List[Dict[str, Union[str, List[str]]]]]:
    """
    Convert features_mapping format to multilevel cupac_features format.
    
    Args:
        features_mapping: Dict with format:
            {
                "target_column": {
                    ("pre_target_column", period): ["covariate1", "covariate2", ...],
                    ...
                }
            }
    
    Returns:
        multilevel_cupac_features: Dict with format:
            {
                "target_column": [
                    {
                        "train_period": 3,
                        "predict_period": 2, 
                        "train_target": "target_lag_3",
                        "predict_target": "target_lag_2",
                        "covariates": ["cov1_lag3", "cov2_lag3"]
                    },
                    {
                        "train_period": 2,
                        "predict_period": 1,
                        "train_target": "target_lag_2", 
                        "predict_target": "target_lag_1",
                        "covariates": ["cov1_lag2", "cov2_lag2"]
                    },
                    ...
                ]
            }
    
    Logic:
        - Creates models for each available lag transition: period N → period N-1
        - Each model learns to predict period N-1 target from period N covariates
        - Final prediction applies sequentially: period 3→2→1→0 (current)
    """
    multilevel_cupac_features = {}
    
    for target_column, mappings in features_mapping.items():
        if not mappings:
            warnings.warn(f"Empty mappings for target column {target_column}")
            continue
            
        # Sort periods in descending order
        sorted_periods = sorted(mappings.items(), key=lambda x: x[0][1], reverse=True)
        
        # Create models for each transition
        models_config = []
        
        for i in range(len(sorted_periods) - 1):
            # Current period (higher) -> next period (lower)
            current_key, current_covariates = sorted_periods[i]
            next_key, next_covariates = sorted_periods[i + 1]
            
            current_target, current_period = current_key
            next_target, next_period = next_key
            
            models_config.append({
                "train_period": current_period,
                "predict_period": next_period,
                "train_target": current_target,
                "predict_target": next_target,
                "covariates": current_covariates
            })
        
        # Add final model: period 1 -> current target
        if sorted_periods:
            final_key, final_covariates = sorted_periods[-1]  # Smallest period
            final_target, final_period = final_key
            
            models_config.append({
                "train_period": final_period,
                "predict_period": 0,  # Current target
                "train_target": final_target,
                "predict_target": target_column,
                "covariates": final_covariates
            })
        
        multilevel_cupac_features[target_column] = models_config
    
    return multilevel_cupac_features


def features_mapping_to_cupac_features(
    features_mapping: Dict[str, Dict[Tuple[str, int], List[str]]]
) -> Dict[str, Dict[str, Union[str, List[str]]]]:
    """
    Convert features_mapping format to cupac_features format for backward compatibility.
    
    This is a simplified version that uses only the highest period for training,
    mainly used for legacy compatibility.
    """
    cupac_features = {}
    
    for target_column, mappings in features_mapping.items():
        if not mappings:
            warnings.warn(f"Empty mappings for target column {target_column}")
            continue
            
        # Find the mapping with the highest period for training
        max_period_mapping = max(mappings.items(), key=lambda x: x[0][1])
        pre_target_column, max_period = max_period_mapping[0]
        training_covariates = max_period_mapping[1]
        
        cupac_features[target_column] = {
            "pre_target": pre_target_column,
            "covariates": training_covariates
        }
    
    return cupac_features


def get_inference_covariates_from_features_mapping(
    features_mapping: Dict[str, Dict[Tuple[str, int], List[str]]],
    target_column: str
) -> List[str]:
    """
    Get covariates for inference (period 1) from features_mapping.
    
    Args:
        features_mapping: Features mapping dictionary
        target_column: Target column name
        
    Returns:
        List of covariate column names for inference (period 1)
    """
    if target_column not in features_mapping:
        return []
    
    mappings = features_mapping[target_column]
    
    # Find period 1 mapping for inference
    period_1_mappings = [
        (key, covariates) 
        for key, covariates in mappings.items() 
        if key[1] == 1
    ]
    
    if not period_1_mappings:
        warnings.warn(f"No period 1 mapping found for target column {target_column}")
        return []
    
    # Use the first period 1 mapping found
    return period_1_mappings[0][1]


def validate_features_mapping(
    features_mapping: Dict[str, Dict[Tuple[str, int], List[str]]]
) -> bool:
    """
    Validate features_mapping format and consistency.
    
    Args:
        features_mapping: Features mapping to validate
        
    Returns:
        True if valid, raises ValueError if not
    """
    for target_column, mappings in features_mapping.items():
        if not isinstance(target_column, str):
            raise ValueError(f"Target column must be string, got {type(target_column)}")
        
        if not isinstance(mappings, dict):
            raise ValueError(f"Mappings for {target_column} must be dict")
        
        if not mappings:
            raise ValueError(f"Empty mappings for target column {target_column}")
        
        # Check covariate length consistency
        covariate_lengths = set()
        for key, covariates in mappings.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(f"Mapping key must be tuple of (str, int), got {key}")
            
            pre_target_col, period = key
            if not isinstance(pre_target_col, str):
                raise ValueError(f"Pre-target column must be string, got {type(pre_target_col)}")
            
            if not isinstance(period, int) or period < 1:
                raise ValueError(f"Period must be positive integer, got {period}")
            
            if not isinstance(covariates, list):
                raise ValueError(f"Covariates must be list, got {type(covariates)}")
            
            if not all(isinstance(cov, str) for cov in covariates):
                raise ValueError(f"All covariates must be strings")
            
            covariate_lengths.add(len(covariates))
        
        # All periods must have same number of covariates
        if len(covariate_lengths) > 1:
            raise ValueError(
                f"All periods for target '{target_column}' must have same number of covariates. "
                f"Found lengths: {sorted(covariate_lengths)}"
            )
    
    return True


class CupacExtension(MLExtension):
    """
    Extension for CUPAC variance reduction with features_mapping support.
    
    This extension can work in two modes:
    1. Traditional mode: uses cupac_features format (same covariates for training and inference)
    2. Enhanced mode: uses features_mapping to train on higher periods and infer on period 1
    """
    def __init__(
        self,
        cupac_features: Dict[str, Dict[str, Union[str, Sequence[str]]]] = None,
        available_models: Dict[str, Any] = None,
        explicit_models: list[str] = None,
        n_folds: int = 5,
        random_state: Optional[int] = None,
        features_mapping: Dict[str, Dict[tuple[str, int], list[str]]] | None = None,
    ):
        super().__init__()
        self.cupac_features = cupac_features or {}
        self.available_models = available_models or {}
        self.explicit_models = explicit_models or []
        self.n_folds = n_folds
        self.random_state = random_state
        self.features_mapping = features_mapping or {}
        self.fitted_models = {}
        self.best_model_names = {}
        self.variance_reductions = {}
        self.is_fitted = False

    def _calc_pandas(
        self,
        data: Dataset,
        mode: str = "auto",
        **kwargs,
    ):
        """Pandas-specific implementation of enhanced CUPAC."""
        if mode in ["auto", "fit"]:
            return self._fit_pandas(data)
        elif mode == "predict":
            return self._predict_pandas(data)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def extract_cupac_from_features_mapping(self, data: Dataset):
        """Extract CUPAC configuration from dataset features_mapping."""
        if hasattr(data, 'features_mapping') and data.features_mapping:
            validate_features_mapping(data.features_mapping)
            # Only update if not already set in constructor
            if not self.features_mapping:
                self.features_mapping = data.features_mapping
            self.cupac_features = features_mapping_to_cupac_features(data.features_mapping)
            return True
        return False

    def _fit_pandas(self, data: Dataset):
        """Fit multilevel models using pandas backend."""
        df = data.data.copy()
        
        self.fitted_models = {}
        self.best_model_names = {}
        
        # Check if we should use multilevel approach
        if self.features_mapping:
            multilevel_config = create_multilevel_cupac_features(self.features_mapping)
            return self._fit_multilevel_pandas(df, multilevel_config)
        else:
            # Fall back to traditional single-level approach
            return self._fit_single_level_pandas(df)
    
    def _fit_multilevel_pandas(self, df, multilevel_config):
        """Fit ONE multilevel model per target using combined data from all period transitions."""
        for target_feature, models_config in multilevel_config.items():
            
            # Combine data from all period transitions into one training dataset
            # We need to standardize column names across periods for concatenation
            X_rows = []
            y_rows = []
            
            for level_config in models_config:
                train_target = level_config["train_target"]
                predict_target = level_config["predict_target"]
                covariates = level_config["covariates"]
                
                # Create standardized feature names: [cov1, cov2, ..., pre_target]
                if len(covariates) == 0:
                    X_level = df[[train_target]].rename(columns={train_target: 'pre_target'})
                    standardized_cols = ['pre_target']
                else:
                    # Rename covariates to standardized names
                    X_level = df[covariates + [train_target]].copy()
                    col_mapping = {}
                    for i, cov in enumerate(covariates):
                        col_mapping[cov] = f'cov_{i}'
                    col_mapping[train_target] = 'pre_target'
                    X_level = X_level.rename(columns=col_mapping)
                    standardized_cols = [f'cov_{i}' for i in range(len(covariates))] + ['pre_target']
                
                # Target to predict
                if level_config["predict_period"] == 0:
                    y_level = df[predict_target]  # Current target
                else:
                    y_level = df[predict_target]  # Previous period target
                
                X_rows.append(X_level)
                y_rows.append(y_level)
            
            # Concatenate all transitions into one dataset with consistent columns
            X_train = pd.concat(X_rows, axis=0, ignore_index=True)
            y_train = pd.concat(y_rows, axis=0, ignore_index=True)
            
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            if len(self.explicit_models) == 1:
                model_name = self.explicit_models[0]
                model_proto = self.available_models[model_name]
                model = clone(model_proto)
                model.fit(X_train, y_train)
                
                # Store single model for this target
                self.fitted_models[target_feature] = model
                self.best_model_names[target_feature] = model_name
                continue
            
            best_score = -np.inf
            best_model = None
            best_model_name = None
            
            for name in self.explicit_models:
                model_proto = self.available_models[name]
                fold_scores = []
                
                for train_idx, val_idx in kf.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    m = clone(model_proto)
                    m.fit(X_tr, y_tr)
                    pred = m.predict(X_val)
                    
                    # Calculate score (negative MSE for maximization)
                    score = -np.mean((y_val - pred) ** 2)
                    fold_scores.append(score)
                
                avg_score = float(np.mean(fold_scores))
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = clone(model_proto)
                    best_model_name = name
            
            if best_model is None:
                raise RuntimeError(f"No model was selected for {target_feature}")
            
            best_model.fit(X_train, y_train)
            
            # Store single model for this target
            self.fitted_models[target_feature] = best_model
            self.best_model_names[target_feature] = best_model_name
        
        self.is_fitted = True
        return self
    
    def _fit_single_level_pandas(self, df):
        """Fit traditional single-level models (backward compatibility)."""
        for target_feature, config in self.cupac_features.items():            
            pre_target = config["pre_target"]
            covariates = config["covariates"]
            
            if len(covariates) == 0:
                X_cov = df[[pre_target]]
            else:
                X_cov = df[covariates]
            y_pre = df[pre_target]  # Predict pre-experiment target
            
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            if len(covariates) == 0:
                X_cov_full = df[[pre_target]]
            else:
                X_cov_full = df[covariates]
            
            if len(self.explicit_models) == 1:
                model_name = self.explicit_models[0]
                model_proto = self.available_models[model_name]
                model = clone(model_proto)
                model.fit(X_cov_full, y_pre)
                self.fitted_models[target_feature] = model
                self.best_model_names[target_feature] = model_name
                # Calculate variance reduction for the model
                pred = model.predict(X_cov_full)
                y_current = df[target_feature]
                y_adj = y_current - pred + y_pre.mean()
                self.variance_reductions[target_feature] = self._calculate_variance_reduction(y_current.to_numpy(), y_adj.to_numpy())
                continue
                
            best_score = -np.inf
            best_model = None
            best_model_name = None
            
            for name in self.explicit_models:
                model_proto = self.available_models[name]
                fold_var_reductions = []
                for train_idx, val_idx in kf.split(X_cov_full):
                    X_train, X_val = X_cov_full.iloc[train_idx], X_cov_full.iloc[val_idx]
                    y_train, y_val = y_pre.iloc[train_idx], y_pre.iloc[val_idx]
                    y_current_val = df[target_feature].iloc[val_idx]
                    
                    m = clone(model_proto)
                    m.fit(X_train, y_train)
                    pred = m.predict(X_val)
                    y_adj = y_current_val - pred + y_train.mean()
                    fold_var_reductions.append(self._calculate_variance_reduction(y_current_val.to_numpy(), y_adj.to_numpy()))
                score = float(np.nanmean(fold_var_reductions))
                if score > best_score:
                    best_score = score
                    best_model = clone(model_proto)
                    best_model_name = name
                    
            if best_model is None:
                raise RuntimeError("No model was selected during model search")
            best_model.fit(X_cov_full, y_pre)
            self.fitted_models[target_feature] = best_model
            self.best_model_names[target_feature] = best_model_name
            # Calculate variance reduction for the best model
            pred = best_model.predict(X_cov_full)
            y_current = df[target_feature]
            y_adj = y_current - pred + y_pre.mean()
            self.variance_reductions[target_feature] = self._calculate_variance_reduction(y_current.to_numpy(), y_adj.to_numpy())
            
        self.is_fitted = True
        return self

    def _predict_pandas(self, data: Dataset) -> Dict[str, Any]:
        """Make predictions using pandas backend with multilevel support."""
        df = data.data.copy()
        result = {}
        
        # Check if we have multilevel models
        if self.features_mapping:
            return self._predict_multilevel_pandas(df)
        else:
            return self._predict_single_level_pandas(df)
    
    def _predict_multilevel_pandas(self, df) -> Dict[str, Any]:
        """Make predictions using ONE multilevel model per target - inference on period 1 data."""
        result = {}
        
        for target_feature, model in self.fitted_models.items():
            if not hasattr(model, 'predict'):
                # Fall back to single-level prediction if not a sklearn model
                continue
            
            # Get inference covariates (period 1 data from features_mapping)
            if hasattr(self, 'features_mapping') and self.features_mapping:
                inference_covariates = get_inference_covariates_from_features_mapping(
                    self.features_mapping, target_feature
                )
                
                if not inference_covariates:
                    continue
                
                # Find period 1 target
                mappings = self.features_mapping[target_feature]
                period_1_mapping = None
                for (pre_target, period), covs in mappings.items():
                    if period == 1:
                        period_1_mapping = (pre_target, covs)
                        break
                
                if period_1_mapping is None:
                    continue
                
                pre_target, covariates = period_1_mapping
                
                # Prepare input with standardized column names (same format as training)
                if len(covariates) == 0:
                    X_input = df[[pre_target]].rename(columns={pre_target: 'pre_target'})
                else:
                    X_input = df[covariates + [pre_target]].copy()
                    col_mapping = {}
                    for i, cov in enumerate(covariates):
                        col_mapping[cov] = f'cov_{i}'
                    col_mapping[pre_target] = 'pre_target'
                    X_input = X_input.rename(columns=col_mapping)
                
                # Make prediction for current target
                prediction = model.predict(X_input)
                
                # Apply CUPAC correction: y_cupac = y - prediction + y_mean
                y_current = df[target_feature]
                y_mean = y_current.mean()
                
                y_adjusted = y_current - prediction + y_mean
                result[f"{target_feature}_cupac"] = y_adjusted
                
                # Calculate variance reduction
                var_reduction = self._calculate_variance_reduction(
                    y_current.to_numpy(), 
                    y_adjusted.to_numpy()
                )
                self.variance_reductions[target_feature] = var_reduction
        
        return result
    
    def _predict_single_level_pandas(self, df) -> Dict[str, Any]:
        """Make predictions using single-level models (backward compatibility)."""
        result = {}
        
        for target_feature, config in self.cupac_features.items():
            pre_target = config["pre_target"]
            covariates = config["covariates"]
            
            model = self.fitted_models.get(target_feature)
            if model is None:
                raise RuntimeError(f"Model for {target_feature} not fitted. Call fit() first.")
            
            # Check if we have features_mapping for enhanced inference
            if self.features_mapping and target_feature in self.features_mapping:
                # Get inference covariates (period 1) from features_mapping
                inference_covariates = get_inference_covariates_from_features_mapping(
                    self.features_mapping, target_feature
                )
                
                if inference_covariates:
                    try:
                        X_cov = df[inference_covariates]
                        # Rename columns to match training features for sklearn compatibility
                        X_cov.columns = covariates
                    except KeyError as e:
                        warnings.warn(f"Inference covariates {inference_covariates} not found in data. "
                                    f"Falling back to training covariates. Missing columns: {e}")
                        # Fall back to training covariates
                        if len(covariates) == 0:
                            X_cov = df[[pre_target]]
                        else:
                            X_cov = df[covariates]
                else:
                    # No period 1 covariates found, use training covariates
                    if len(covariates) == 0:
                        X_cov = df[[pre_target]]
                    else:
                        X_cov = df[covariates]
            else:
                # Traditional mode: use same covariates for training and inference
                if len(covariates) == 0:
                    X_cov = df[[pre_target]]
                else:
                    X_cov = df[covariates]
            
            y_current = df[target_feature]  # Current experiment target
            y_pre_mean = df[pre_target].mean()  # Mean of pre-experiment target
            
            # Predict pre-experiment target using covariates
            pred_pre = model.predict(X_cov)
            
            # Adjust current target by subtracting prediction and adding back the mean
            y_adj = y_current - pred_pre + y_pre_mean
            result[f"{target_feature}_cupac"] = y_adj
        
        return result

    def fit(self, X: Dataset, Y: Dataset = None) -> 'CupacExtension':
        return super().calc(X, mode="fit")

    def predict(self, X: Dataset) -> Dict[str, Any]:
        return super().calc(X, mode="predict")

    def calc(self, data: Dataset, **kwargs):
        self.fit(data)
        return self.predict(data)

    def get_variance_reductions(self):
        return {f"{target}_cupac_variance_reduction": reduction for target, reduction in self.variance_reductions.items()}

    @staticmethod
    def _calculate_variance_reduction(y_original, y_adjusted):
        """Calculate variance reduction between original and adjusted target."""
        var_original = np.var(y_original)
        var_adjusted = np.var(y_adjusted)
        if var_original < 1e-10:
            return 0.0
        return float(max(0, (1 - var_adjusted / var_original) * 100))
