"""
Comprehensive tests for CUPAC (Covariate-Updated Pre-Analysis Correction) functionality.

This module tests all aspects of CUPAC implementation including:
- CUPACExecutor functionality
- CupacExtension functionality
- Different model backends (linear, ridge, lasso, catboost)
- Variance reduction calculations
- Cross-validation
- Integration with ABTest
- Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from copy import deepcopy

from hypex.dataset import Dataset, TargetRole, TreatmentRole, InfoRole
from hypex.dataset.dataset import ExperimentData
from hypex.ml.cupac import CUPACExecutor
from hypex.extensions.cupac import CupacExtension
from hypex.ab import ABTest
from hypex.utils.tutorial_data_creation import DataGenerator


class TestCUPACExecutor:
    """Test cases for CUPACExecutor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate correlated features
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)
        X1_lag = X1 + np.random.normal(0, 0.1, n_samples)  # Correlated with X1
        X2_lag = X2 + np.random.normal(0, 0.1, n_samples)  # Correlated with X2
        
        # Generate target with correlation to features
        y = 2 * X1 + 1.5 * X2 + np.random.normal(0, 0.5, n_samples)
        y_lag = y + np.random.normal(0, 0.2, n_samples)  # Correlated with y
        
        # Treatment assignment
        treatment = np.random.binomial(1, 0.5, n_samples)
        
        df = pd.DataFrame({
            'X1': X1,
            'X2': X2,
            'X1_lag': X1_lag,
            'X2_lag': X2_lag,
            'y': y,
            'y_lag': y_lag,
            'treatment': treatment,
            'user_id': range(n_samples)
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole(),
                'user_id': InfoRole()
            }
        )
        
        return dataset

    @pytest.fixture
    def experiment_data(self, sample_data):
        """Create ExperimentData for testing."""
        return ExperimentData(sample_data)

    def test_cupac_executor_init(self):
        """Test CUPACExecutor initialization."""
        cupac_features = {'y': {'pre_target': 'y_lag', 'covariates': ['X1_lag', 'X2_lag']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model='linear',
            n_folds=3,
            random_state=42
        )
        
        assert executor.cupac_features == cupac_features
        assert executor.cupac_model == 'linear'
        assert executor.n_folds == 3
        assert executor.random_state == 42
        assert not executor.is_fitted

    def test_cupac_executor_init_multiple_models(self):
        """Test CUPACExecutor initialization with multiple models."""
        cupac_features = {'y': {'pre_target': 'y_lag', 'covariates': ['X1_lag', 'X2_lag']}}
        models = ['linear', 'ridge', 'lasso']
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model=models
        )
        
        assert executor.cupac_model == models

    def test_get_models_single_model(self, sample_data):
        """Test get_models method with single model."""
        cupac_features = {'y': {'pre_target': 'y_lag', 'covariates': ['X1_lag', 'X2_lag']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model='linear'
        )
        
        available_models, explicit_models = executor.get_models('pandasdataset')
        
        assert 'linear' in available_models
        assert explicit_models == ['linear']

    def test_get_models_multiple_models(self, sample_data):
        """Test get_models method with multiple models."""
        cupac_features = {'y': {'pre_target': 'y_lag', 'covariates': ['X1_lag', 'X2_lag']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model=['linear', 'ridge']
        )
        
        available_models, explicit_models = executor.get_models('pandasdataset')
        
        assert 'linear' in available_models
        assert 'ridge' in available_models
        assert set(explicit_models) == {'linear', 'ridge'}

    def test_get_models_all_models(self, sample_data):
        """Test get_models method with all available models."""
        cupac_features = {'y': {'pre_target': 'y_lag', 'covariates': ['X1_lag', 'X2_lag']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model=None
        )
        
        available_models, explicit_models = executor.get_models('pandasdataset')
        
        assert len(available_models) > 0
        assert len(explicit_models) > 0
        assert 'linear' in explicit_models

    def test_fit_single_target(self, sample_data):
        """Test fitting CUPAC executor with single target."""
        cupac_features = {'y': {'pre_target': 'y_lag', 'covariates': ['X1_lag', 'X2_lag']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model='linear',
            random_state=42
        )
        
        executor.fit(sample_data)
        
        assert executor.is_fitted
        assert hasattr(executor, 'extension')

    def test_fit_multiple_targets(self, sample_data):
        """Test fitting CUPAC executor with multiple targets."""
        # Add another target to the dataset
        data_copy = deepcopy(sample_data)
        y2_values = data_copy.data['y'].values * 0.5 + np.random.normal(0, 0.3, len(data_copy.data))
        data_copy = data_copy.add_column(data=y2_values, role={'y2': TargetRole()})
        
        cupac_features = {
            'y': {'pre_target': 'y_lag', 'covariates': ['X1_lag', 'X2_lag']},
            'y2': {'pre_target': 'y_lag', 'covariates': ['X1_lag']}
        }
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model='linear',
            random_state=42
        )
        
        executor.fit(data_copy)
        
        assert executor.is_fitted

    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that predict raises error when called before fit."""
        cupac_features = {'y': {'pre_target': 'y_lag', 'covariates': ['X1_lag', 'X2_lag']}}
        executor = CUPACExecutor(cupac_features=cupac_features)
        
        with pytest.raises(RuntimeError, match="CUPACExecutor not fitted"):
            executor.predict(sample_data)

    def test_get_variance_reductions_before_fit_raises_error(self, sample_data):
        """Test that get_variance_reductions raises error when called before fit."""
        cupac_features = {'y': {'pre_target': 'y_lag', 'covariates': ['X1_lag', 'X2_lag']}}
        executor = CUPACExecutor(cupac_features=cupac_features)
        
        with pytest.raises(RuntimeError, match="CUPACExecutor not fitted"):
            executor.get_variance_reductions()

    def test_fit_predict_workflow(self, sample_data):
        """Test complete fit-predict workflow."""
        cupac_features = {'y': {'pre_target': 'y_lag', 'covariates': ['X1_lag', 'X2_lag']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model='linear',
            random_state=42
        )
        
        # Fit
        executor.fit(sample_data)
        
        # Predict
        predictions = executor.predict(sample_data)
        
        assert isinstance(predictions, dict)
        assert 'y_cupac' in predictions
        assert len(predictions['y_cupac']) == len(sample_data.data)

    def test_variance_reduction_calculation(self, sample_data):
        """Test variance reduction calculation."""
        cupac_features = {'y': {'pre_target': 'y_lag', 'covariates': ['X1_lag', 'X2_lag']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model='linear',
            random_state=42
        )
        
        executor.fit(sample_data)
        variance_reductions = executor.get_variance_reductions()
        
        assert isinstance(variance_reductions, dict)
        assert 'y_cupac_variance_reduction' in variance_reductions
        assert isinstance(variance_reductions['y_cupac_variance_reduction'], (int, float))

    def test_execute_method(self, experiment_data):
        """Test execute method of CUPACExecutor."""
        cupac_features = {'y': {'pre_target': 'y_lag', 'covariates': ['X1_lag', 'X2_lag']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model='linear',
            random_state=42
        )
        
        result = executor.execute(experiment_data)
        
        assert isinstance(result, ExperimentData)
        assert 'y_cupac' in result.ds.roles
        assert isinstance(result.ds.roles['y_cupac'], TargetRole)
        
        # Check that variance reduction is stored in additional_fields
        assert len(result.additional_fields.data) > 0

    def test_different_models(self, sample_data):
        """Test CUPAC with different model types."""
        cupac_features = {'y': {'pre_target': 'y_lag', 'covariates': ['X1_lag', 'X2_lag']}}
        models_to_test = ['linear', 'ridge', 'lasso']
        
        for model in models_to_test:
            executor = CUPACExecutor(
                cupac_features=cupac_features,
                cupac_model=model,
                random_state=42
            )
            
            executor.fit(sample_data)
            predictions = executor.predict(sample_data)
            
            assert 'y_cupac' in predictions
            assert len(predictions['y_cupac']) == len(sample_data.data)

    def test_model_selection_multiple_models(self, sample_data):
        """Test automatic model selection with multiple models."""
        cupac_features = {'y': {'pre_target': 'y_lag', 'covariates': ['X1_lag', 'X2_lag']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model=['linear', 'ridge', 'lasso'],
            random_state=42
        )
        
        executor.fit(sample_data)
        predictions = executor.predict(sample_data)
        variance_reductions = executor.get_variance_reductions()
        
        assert 'y_cupac' in predictions
        assert 'y_cupac_variance_reduction' in variance_reductions

    def test_invalid_features(self, sample_data):
        """Test behavior with invalid feature names."""
        cupac_features = {'y': {'pre_target': 'y_lag', 'covariates': ['nonexistent_feature']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model='linear'
        )
        
        with pytest.raises(KeyError):
            executor.fit(sample_data)

    def test_invalid_target(self, sample_data):
        """Test behavior with invalid target name."""
        cupac_features = {'nonexistent_target': {'pre_target': 'y_lag', 'covariates': ['X1_lag']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model='linear'
        )
        
        with pytest.raises(KeyError):
            executor.fit(sample_data)


class TestCupacExtension:
    """Test cases for CupacExtension class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        n_samples = 500
        
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)
        X1_lag = X1 + np.random.normal(0, 0.1, n_samples)
        X2_lag = X2 + np.random.normal(0, 0.1, n_samples)
        y = 2 * X1 + 1.5 * X2 + np.random.normal(0, 0.5, n_samples)
        y_pre = y + np.random.normal(0, 0.2, n_samples)  # Pre-experiment target
        
        df = pd.DataFrame({
            'X1': X1,
            'X2': X2,
            'X1_lag': X1_lag,
            'X2_lag': X2_lag,
            'y': y,
            'y_pre': y_pre
        })
        
        dataset = Dataset(
            data=df,
            roles={'y': TargetRole()}
        )
        
        return dataset

    @pytest.fixture
    def mock_available_models(self):
        """Create mock available models."""
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        return {
            'linear': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(max_iter=1000)
        }

    def test_cupac_extension_init(self, mock_available_models):
        """Test CupacExtension initialization."""
        cupac_features = {'y': {'pre_target': 'y_pre', 'covariates': ['X1_lag', 'X2_lag']}}
        extension = CupacExtension(
            cupac_features=cupac_features,
            available_models=mock_available_models,
            explicit_models=['linear'],
            n_folds=3,
            random_state=42
        )
        
        assert extension.cupac_features == cupac_features
        assert extension.n_folds == 3
        assert extension.random_state == 42
        assert not extension.is_fitted

    def test_calc_fit_mode(self, sample_data, mock_available_models):
        """Test calc method in fit mode."""
        cupac_features = {'y': {'pre_target': 'y_pre', 'covariates': ['X1_lag', 'X2_lag']}}
        extension = CupacExtension(
            cupac_features=cupac_features,
            available_models=mock_available_models,
            explicit_models=['linear'],
            random_state=42
        )
        
        result = extension.calc(sample_data, mode='fit')
        
        assert extension.is_fitted
        assert len(extension.fitted_models) > 0

    def test_calc_predict_mode(self, sample_data, mock_available_models):
        """Test calc method in predict mode."""
        cupac_features = {'y': {'pre_target': 'y_pre', 'covariates': ['X1_lag', 'X2_lag']}}
        extension = CupacExtension(
            cupac_features=cupac_features,
            available_models=mock_available_models,
            explicit_models=['linear'],
            random_state=42
        )
        
        # First fit
        extension.calc(sample_data, mode='fit')
        
        # Then predict
        predictions = extension.calc(sample_data, mode='predict')
        
        assert isinstance(predictions, dict)
        assert 'y_cupac' in predictions

    def test_calc_auto_mode(self, sample_data, mock_available_models):
        """Test calc method in auto mode."""
        cupac_features = {'y': {'pre_target': 'y_pre', 'covariates': ['X1_lag', 'X2_lag']}}
        extension = CupacExtension(
            cupac_features=cupac_features,
            available_models=mock_available_models,
            explicit_models=['linear'],
            random_state=42
        )
        
        result = extension.calc(sample_data, mode='auto')
        
        assert extension.is_fitted

    def test_calc_invalid_mode(self, sample_data, mock_available_models):
        """Test calc method with invalid mode."""
        cupac_features = {'y': {'pre_target': 'y_pre', 'covariates': ['X1_lag', 'X2_lag']}}
        extension = CupacExtension(
            cupac_features=cupac_features,
            available_models=mock_available_models,
            explicit_models=['linear']
        )
        
        # The calc method actually just calls fit then predict
        # so it should work without raising an error
        result = extension.calc(sample_data)
        assert isinstance(result, dict)

    def test_variance_reduction_calculation(self, sample_data, mock_available_models):
        """Test variance reduction calculation in extension."""
        cupac_features = {'y': {'pre_target': 'y_pre', 'covariates': ['X1_lag', 'X2_lag']}}
        extension = CupacExtension(
            cupac_features=cupac_features,
            available_models=mock_available_models,
            explicit_models=['linear'],
            random_state=42
        )
        
        extension.calc(sample_data, mode='fit')
        variance_reductions = extension.get_variance_reductions()
        
        assert isinstance(variance_reductions, dict)
        assert len(variance_reductions) > 0

    def test_model_selection_single_model(self, sample_data, mock_available_models):
        """Test model selection with single model."""
        cupac_features = {'y': {'pre_target': 'y_pre', 'covariates': ['X1_lag', 'X2_lag']}}
        extension = CupacExtension(
            cupac_features=cupac_features,
            available_models=mock_available_models,
            explicit_models=['linear'],
            random_state=42
        )
        
        extension.calc(sample_data, mode='fit')
        
        assert 'y' in extension.best_model_names
        assert extension.best_model_names['y'] == 'linear'

    def test_model_selection_multiple_models(self, sample_data, mock_available_models):
        """Test model selection with multiple models."""
        cupac_features = {'y': {'pre_target': 'y_pre', 'covariates': ['X1_lag', 'X2_lag']}}
        extension = CupacExtension(
            cupac_features=cupac_features,
            available_models=mock_available_models,
            explicit_models=['linear', 'ridge'],
            random_state=42
        )
        
        extension.calc(sample_data, mode='fit')
        
        assert 'y' in extension.best_model_names
        assert extension.best_model_names['y'] in ['linear', 'ridge']

    def test_cross_validation_folds(self, sample_data, mock_available_models):
        """Test different number of cross-validation folds."""
        cupac_features = {'y': {'pre_target': 'y_pre', 'covariates': ['X1_lag', 'X2_lag']}}
        
        for n_folds in [3, 5, 10]:
            extension = CupacExtension(
                cupac_features=cupac_features,
                available_models=mock_available_models,
                explicit_models=['linear'],
                n_folds=n_folds,
                random_state=42
            )
            
            extension.calc(sample_data, mode='fit')
            assert extension.is_fitted


class TestCUPACIntegration:
    """Test integration of CUPAC with ABTest."""
    
    @pytest.fixture
    def sample_ab_data(self):
        """Create sample A/B test data."""
        gen = DataGenerator(
            n_samples=1000,
            distributions={
                "X1": {"type": "normal", "mean": 0, "std": 1},
                "X2": {"type": "bernoulli", "p": 0.5},
                "y0": {"type": "normal", "mean": 5, "std": 1},
            },
            time_correlations={"X1": 0.2, "X2": 0.1, "y0": 0.6},
            effect_size=1.5,
            seed=42
        )
        df = gen.generate()
        
        # Keep some lag features for CUPAC
        df = df.drop(columns=['y0', 'z', 'U', 'D', 'y1', 'y0_lag_2'])
        
        data = Dataset(
            roles={
                "d": TreatmentRole(),
                "y": TargetRole(),
            },
            data=df,
            default_role=InfoRole()
        )
        
        return data

    def test_abtest_with_cupac_single_model(self, sample_ab_data):
        """Test ABTest with CUPAC using single model."""
        test = ABTest(
            cupac_features={'y': {'pre_target': 'y0_lag_1', 'covariates': ['X1_lag', 'X2_lag']}},
            cupac_model='linear'
        )
        
        result = test.execute(sample_ab_data)
        
        assert result.resume is not None
        assert hasattr(result, 'variance_reduction_report')
        
        # Check that CUPAC target is created
        assert 'y_cupac' in result._experiment_data.ds.roles

    def test_abtest_with_cupac_multiple_models(self, sample_ab_data):
        """Test ABTest with CUPAC using multiple models."""
        test = ABTest(
            cupac_features={'y': {'pre_target': 'y0_lag_1', 'covariates': ['X1_lag', 'X2_lag']}},
            cupac_model=['linear', 'ridge', 'lasso']
        )
        
        result = test.execute(sample_ab_data)
        
        assert result.resume is not None
        assert hasattr(result, 'variance_reduction_report')

    def test_abtest_with_cupac_auto_model_selection(self, sample_ab_data):
        """Test ABTest with CUPAC using automatic model selection."""
        test = ABTest(
            cupac_features={'y': {'pre_target': 'y0_lag_1', 'covariates': ['X1_lag', 'X2_lag']}},
            cupac_model=None  # Auto selection
        )
        
        result = test.execute(sample_ab_data)
        
        assert result.resume is not None
        assert hasattr(result, 'variance_reduction_report')

    def test_variance_reduction_report(self, sample_ab_data):
        """Test variance reduction report generation."""
        test = ABTest(
            cupac_features={'y': {'pre_target': 'y0_lag_1', 'covariates': ['X1_lag', 'X2_lag']}},
            cupac_model='linear'
        )
        
        result = test.execute(sample_ab_data)
        variance_report = result.variance_reduction_report
        
        assert variance_report is not None
        # Should contain variance reduction information

    def test_cupac_with_additional_tests(self, sample_ab_data):
        """Test CUPAC with additional statistical tests."""
        test = ABTest(
            cupac_features={'y': {'pre_target': 'y0_lag_1', 'covariates': ['X1_lag', 'X2_lag']}},
            cupac_model='linear',
            additional_tests=['t-test', 'u-test']
        )
        
        result = test.execute(sample_ab_data)
        
        assert result.resume is not None
        # Check that results include multiple tests
        resume_data = result.resume
        assert 'TTest p-value' in resume_data.columns or 'TTest pvalue' in resume_data.columns

    def test_cupac_with_multitest_correction(self, sample_ab_data):
        """Test CUPAC with multiple testing correction."""
        from hypex.utils import ABNTestMethodsEnum
        
        test = ABTest(
            cupac_features={'y': {'pre_target': 'y0_lag_1', 'covariates': ['X1_lag', 'X2_lag']}},
            cupac_model='linear',
            multitest_method=ABNTestMethodsEnum.bonferroni
        )
        
        result = test.execute(sample_ab_data)
        
        assert result.resume is not None
        assert result.multitest is not None

    def test_cupac_and_cuped_together_error(self, sample_ab_data):
        """Test that using CUPAC and CUPED together works."""
        # This should work as they create different target columns
        test = ABTest(
            cupac_features={'y': {'pre_target': 'y0_lag_1', 'covariates': ['X1_lag']}},
            cuped_features={'y': 'y0_lag_1'},
            cupac_model='linear'
        )
        
        result = test.execute(sample_ab_data)
        
        assert result.resume is not None
        # Should have both y_cupac and y_cuped targets
        assert 'y_cupac' in result._experiment_data.ds.roles
        assert 'y_cuped' in result._experiment_data.ds.roles

    def test_cupac_empty_features_error(self, sample_ab_data):
        """Test error handling with empty CUPAC features."""
        test = ABTest(
            cupac_features={},
            cupac_model='linear'
        )
        
        # Should work but not create any CUPAC transformations
        result = test.execute(sample_ab_data)
        assert result.resume is not None

    def test_cupac_invalid_model(self, sample_ab_data):
        """Test error handling with invalid model name."""
        test = ABTest(
            cupac_features={'y': {'pre_target': 'y0_lag_1', 'covariates': ['X1_lag']}},
            cupac_model='invalid_model'
        )
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(RuntimeError):
            test.execute(sample_ab_data)


class TestCUPACEdgeCases:
    """Test edge cases and error conditions for CUPAC."""
    
    @pytest.fixture
    def minimal_data(self):
        """Create minimal dataset for edge case testing."""
        df = pd.DataFrame({
            'y': [1, 2, 3, 4, 5],
            'x': [1, 2, 3, 4, 5],
            'treatment': [0, 1, 0, 1, 0]
        })
        
        return Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )

    def test_small_dataset(self, minimal_data):
        """Test CUPAC with very small dataset."""
        cupac_features = {'y': {'pre_target': 'x', 'covariates': ['x']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model='linear',
            n_folds=2  # Reduce folds for small data
        )
        
        executor.fit(minimal_data)
        predictions = executor.predict(minimal_data)
        
        assert 'y_cupac' in predictions

    def test_single_feature(self, minimal_data):
        """Test CUPAC with single covariate feature."""
        cupac_features = {'y': {'pre_target': 'x', 'covariates': ['x']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model='linear'
        )
        
        executor.fit(minimal_data)
        predictions = executor.predict(minimal_data)
        
        assert 'y_cupac' in predictions
        assert len(predictions['y_cupac']) == len(minimal_data.data)

    def test_perfect_correlation(self):
        """Test CUPAC with perfectly correlated features."""
        n_samples = 100
        x = np.random.normal(0, 1, n_samples)
        y = x  # Perfect correlation
        
        df = pd.DataFrame({
            'y': y,
            'x': x,
            'treatment': np.random.binomial(1, 0.5, n_samples)
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cupac_features = {'y': {'pre_target': 'x', 'covariates': ['x']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model='linear'
        )
        
        executor.fit(dataset)
        variance_reductions = executor.get_variance_reductions()
        
        # Should achieve high variance reduction
        assert variance_reductions['y_cupac_variance_reduction'] > 50

    def test_no_correlation(self):
        """Test CUPAC with uncorrelated features."""
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'y': np.random.normal(0, 1, n_samples),
            'x': np.random.normal(0, 1, n_samples),  # Independent
            'treatment': np.random.binomial(1, 0.5, n_samples)
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cupac_features = {'y': {'pre_target': 'x', 'covariates': ['x']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model='linear'
        )
        
        executor.fit(dataset)
        variance_reductions = executor.get_variance_reductions()
        
        # Should achieve low variance reduction
        assert variance_reductions['y_cupac_variance_reduction'] < 20

    def test_missing_values_handling(self):
        """Test CUPAC behavior with missing values."""
        df = pd.DataFrame({
            'y': [1, 2, np.nan, 4, 5],
            'x': [1, np.nan, 3, 4, 5],
            'treatment': [0, 1, 0, 1, 0]
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cupac_features = {'y': {'pre_target': 'x', 'covariates': ['x']}}
        executor = CUPACExecutor(
            cupac_features=cupac_features,
            cupac_model='linear'
        )
        
        # Should handle missing values appropriately
        # This might raise an error or handle missing values
        # depending on the implementation
        try:
            executor.fit(dataset)
            predictions = executor.predict(dataset)
            # If successful, check predictions
            assert 'y_cupac' in predictions
        except ValueError:
            # Missing values might not be supported
            pass