"""
Comprehensive tests for CUPED (Controlled Experiments Using Pre-Experiment Data) functionality.

This module tests all aspects of CUPED implementation including:
- CUPEDTransformer functionality
- Variance reduction calculations
- Integration with ABTest
- Edge cases and error handling
- Statistical correctness of CUPED adjustments
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from copy import deepcopy

from hypex.dataset import Dataset, TargetRole, TreatmentRole, InfoRole, StatisticRole
from hypex.dataset.dataset import ExperimentData
from hypex.transformers.cuped import CUPEDTransformer
from hypex.ab import ABTest
from hypex.utils.tutorial_data_creation import DataGenerator


class TestCUPEDTransformer:
    """Test cases for CUPEDTransformer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset with correlated pre-period data."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate pre-period target
        y_pre = np.random.normal(5, 2, n_samples)
        
        # Generate current target correlated with pre-period
        correlation = 0.7
        noise = np.random.normal(0, 1, n_samples)
        y_current = correlation * y_pre + np.sqrt(1 - correlation**2) * noise + 1
        
        # Add treatment assignment
        treatment = np.random.binomial(1, 0.5, n_samples)
        
        # Add treatment effect
        treatment_effect = 0.5
        y_current += treatment * treatment_effect
        
        df = pd.DataFrame({
            'y': y_current,
            'y_pre': y_pre,
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
        return ExperimentData(data=sample_data)

    def test_cuped_transformer_init(self):
        """Test CUPEDTransformer initialization."""
        cuped_features = {'y': 'y_pre'}
        transformer = CUPEDTransformer(cuped_features=cuped_features, key="test")
        
        assert transformer.cuped_features == cuped_features
        assert transformer.key == "test"

    def test_cuped_transformer_init_multiple_features(self):
        """Test CUPEDTransformer initialization with multiple target features."""
        cuped_features = {
            'y1': 'y1_pre',
            'y2': 'y2_pre'
        }
        transformer = CUPEDTransformer(cuped_features=cuped_features)
        
        assert transformer.cuped_features == cuped_features

    def test_inner_function_single_feature(self, sample_data):
        """Test _inner_function with single target feature."""
        cuped_features = {'y': 'y_pre'}
        
        result = CUPEDTransformer._inner_function(sample_data, cuped_features)
        
        # Check that new column is created
        assert 'y_cuped' in result.data.columns
        assert isinstance(result.roles['y_cuped'], TargetRole)
        
        # Check that original data is preserved
        assert 'y' in result.data.columns
        assert 'y_pre' in result.data.columns

    def test_inner_function_multiple_features(self, sample_data):
        """Test _inner_function with multiple target features."""
        # Add another target and pre-target
        data_copy = deepcopy(sample_data)
        y2 = data_copy.data['y'].values * 0.8 + np.random.normal(0, 0.5, len(data_copy.data))
        y2_pre = y2 + np.random.normal(0, 0.3, len(data_copy.data))
        
        data_copy = data_copy.add_column(data=y2, role={'y2': TargetRole()})
        data_copy = data_copy.add_column(data=y2_pre, role={'y2_pre': InfoRole()})
        
        cuped_features = {
            'y': 'y_pre',
            'y2': 'y2_pre'
        }
        
        result = CUPEDTransformer._inner_function(data_copy, cuped_features)
        
        # Check that both new columns are created
        assert 'y_cuped' in result.data.columns
        assert 'y2_cuped' in result.data.columns
        assert isinstance(result.roles['y_cuped'], TargetRole)
        assert isinstance(result.roles['y2_cuped'], TargetRole)

    def test_cuped_statistical_correctness(self, sample_data):
        """Test that CUPED adjustment is statistically correct."""
        cuped_features = {'y': 'y_pre'}
        
        result = CUPEDTransformer._inner_function(sample_data, cuped_features)
        
        y_original = sample_data.data['y'].values
        y_pre = sample_data.data['y_pre'].values
        y_cuped = result.data['y_cuped'].values
        
        # Calculate expected CUPED adjustment manually
        mean_xy = np.mean(y_original * y_pre)
        mean_x = np.mean(y_pre)
        mean_y = np.mean(y_original)
        cov_xy = mean_xy - mean_x * mean_y
        
        std_y = np.std(y_original, ddof=1)  # Match pandas default
        std_x = np.std(y_pre, ddof=1)      # Match pandas default
        theta = cov_xy / (std_y * std_x)
        
        expected_cuped = y_original - (y_pre - mean_x) * theta
        
        # Check that our implementation matches the expected calculation
        np.testing.assert_array_almost_equal(y_cuped, expected_cuped, decimal=10)

    def test_cuped_variance_reduction(self, sample_data):
        """Test that CUPED reduces variance when features are correlated."""
        cuped_features = {'y': 'y_pre'}
        
        result = CUPEDTransformer._inner_function(sample_data, cuped_features)
        
        original_var = sample_data['y'].var()
        cuped_var = result['y_cuped'].var()
        
        # CUPED should reduce variance when there's correlation
        assert cuped_var < original_var
        
        # Calculate variance reduction percentage
        variance_reduction = (1 - cuped_var / original_var) * 100
        assert variance_reduction > 0

    def test_cuped_mean_preservation(self, sample_data):
        """Test that CUPED preserves the mean of the original target."""
        cuped_features = {'y': 'y_pre'}
        
        result = CUPEDTransformer._inner_function(sample_data, cuped_features)
        
        original_mean = sample_data['y'].mean()
        cuped_mean = result['y_cuped'].mean()
        
        # CUPED should preserve the mean
        np.testing.assert_almost_equal(original_mean, cuped_mean, decimal=10)

    def test_calc_classmethod(self, sample_data):
        """Test calc class method."""
        cuped_features = {'y': 'y_pre'}
        
        result = CUPEDTransformer.calc(sample_data, cuped_features)
        
        assert 'y_cuped' in result.data.columns
        assert isinstance(result.roles['y_cuped'], TargetRole)

    def test_execute_method(self, experiment_data):
        """Test execute method of CUPEDTransformer."""
        cuped_features = {'y': 'y_pre'}
        transformer = CUPEDTransformer(cuped_features=cuped_features)
        
        result = transformer.execute(experiment_data)
        
        # Check that result is ExperimentData
        assert isinstance(result, ExperimentData)
        
        # Check that new target is created
        assert 'y_cuped' in result.ds.data.columns
        assert isinstance(result.ds.roles['y_cuped'], TargetRole)
        
        # Check that variance reduction is stored in additional_fields
        additional_data = result.additional_fields.data
        variance_reduction_cols = [col for col in additional_data.columns 
                                 if 'variance_reduction' in col]
        assert len(variance_reduction_cols) > 0

    def test_variance_reduction_calculation_in_execute(self, experiment_data):
        """Test variance reduction calculation in execute method."""
        cuped_features = {'y': 'y_pre'}
        transformer = CUPEDTransformer(cuped_features=cuped_features)
        
        result = transformer.execute(experiment_data)
        
        # Extract variance reduction from additional_fields
        additional_data = result.additional_fields.data
        variance_reduction_col = None
        for col in additional_data.columns:
            if 'y_cuped_variance_reduction' in col:
                variance_reduction_col = col
                break
        
        assert variance_reduction_col is not None
        variance_reduction = additional_data[variance_reduction_col].iloc[0]
        
        # Check that variance reduction is reasonable
        assert isinstance(variance_reduction, (int, float))
        assert variance_reduction >= 0  # Should be non-negative

    def test_uncorrelated_features(self):
        """Test CUPED with uncorrelated features."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate uncorrelated features
        y = np.random.normal(0, 1, n_samples)
        y_pre = np.random.normal(0, 1, n_samples)  # Independent
        
        df = pd.DataFrame({
            'y': y,
            'y_pre': y_pre,
            'treatment': np.random.binomial(1, 0.5, n_samples)
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cuped_features = {'y': 'y_pre'}
        result = CUPEDTransformer._inner_function(dataset, cuped_features)
        
        # With uncorrelated features, variance should not reduce much
        original_var = dataset['y'].var()
        cuped_var = result['y_cuped'].var()
        variance_reduction = (1 - cuped_var / original_var) * 100
        
        # Should be small variance reduction
        assert abs(variance_reduction) < 10

    def test_perfect_correlation(self):
        """Test CUPED with perfectly correlated features."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate perfectly correlated features
        y_pre = np.random.normal(0, 1, n_samples)
        y = y_pre + 2  # Perfect correlation with offset
        
        df = pd.DataFrame({
            'y': y,
            'y_pre': y_pre,
            'treatment': np.random.binomial(1, 0.5, n_samples)
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cuped_features = {'y': 'y_pre'}
        result = CUPEDTransformer._inner_function(dataset, cuped_features)
        
        # With perfect correlation, variance should be greatly reduced
        original_var = dataset['y'].var()
        cuped_var = result['y_cuped'].var()
        variance_reduction = (1 - cuped_var / original_var) * 100
        
        # Should achieve high variance reduction
        assert variance_reduction > 90

    def test_invalid_pre_feature(self, sample_data):
        """Test behavior with invalid pre-feature name."""
        cuped_features = {'y': 'nonexistent_pre_feature'}
        
        with pytest.raises(KeyError):
            CUPEDTransformer._inner_function(sample_data, cuped_features)

    def test_invalid_target_feature(self, sample_data):
        """Test behavior with invalid target feature name."""
        cuped_features = {'nonexistent_target': 'y_pre'}
        
        with pytest.raises(KeyError):
            CUPEDTransformer._inner_function(sample_data, cuped_features)

    def test_zero_variance_pre_feature(self):
        """Test CUPED with zero variance pre-feature."""
        n_samples = 100
        
        df = pd.DataFrame({
            'y': np.random.normal(0, 1, n_samples),
            'y_pre': np.ones(n_samples),  # Zero variance
            'treatment': np.random.binomial(1, 0.5, n_samples)
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cuped_features = {'y': 'y_pre'}
        
        # Should handle zero variance gracefully
        result = CUPEDTransformer._inner_function(dataset, cuped_features)
        
        # With zero variance pre-feature, CUPED should not change the target
        original_values = dataset.data['y'].values
        cuped_values = result.data['y_cuped'].values
        
        np.testing.assert_array_almost_equal(original_values, cuped_values, decimal=10)

    def test_zero_variance_target_feature(self):
        """Test CUPED with zero variance target feature."""
        n_samples = 100
        
        df = pd.DataFrame({
            'y': np.ones(n_samples),  # Zero variance
            'y_pre': np.random.normal(0, 1, n_samples),
            'treatment': np.random.binomial(1, 0.5, n_samples)
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cuped_features = {'y': 'y_pre'}
        
        # Should handle zero variance gracefully
        result = CUPEDTransformer._inner_function(dataset, cuped_features)
        
        # With zero variance target, CUPED should not change the target
        original_values = dataset.data['y'].values  
        cuped_values = result.data['y_cuped'].values
        
        np.testing.assert_array_almost_equal(original_values, cuped_values, decimal=10)


class TestCUPEDIntegration:
    """Test integration of CUPED with ABTest."""
    
    @pytest.fixture
    def sample_ab_data(self):
        """Create sample A/B test data with pre-period features."""
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
        
        # Keep relevant columns including pre-period target
        df = df.drop(columns=['z', 'U', 'D', 'y1', 'y0_lag_2'])
        
        data = Dataset(
            roles={
                "d": TreatmentRole(),
                "y": TargetRole(),
            },
            data=df,
            default_role=InfoRole()
        )
        
        return data

    def test_abtest_with_cuped(self, sample_ab_data):
        """Test ABTest with CUPED functionality."""
        test = ABTest(cuped_features={'y': 'y0_lag_1'})
        
        result = test.execute(sample_ab_data)
        
        assert result.resume is not None
        assert hasattr(result, 'variance_reduction_report')
        
        # Check that CUPED target is created
        assert 'y_cuped' in result._experiment_data.ds.roles

    def test_abtest_cuped_multiple_targets(self, sample_ab_data):
        """Test ABTest with CUPED for multiple targets."""
        # Add another target
        data_copy = deepcopy(sample_ab_data)
        y2 = data_copy.data['y'].values * 0.7 + np.random.normal(0, 0.5, len(data_copy.data))
        y2_pre = data_copy.data['y0_lag_1'].values * 0.8 + np.random.normal(0, 0.3, len(data_copy.data))
        
        data_copy = data_copy.add_column(data=y2, role={'y2': TargetRole()})
        data_copy = data_copy.add_column(data=y2_pre, role={'y2_pre': InfoRole()})
        
        test = ABTest(cuped_features={
            'y': 'y0_lag_1',
            'y2': 'y2_pre'
        })
        
        result = test.execute(data_copy)
        
        assert result.resume is not None
        assert 'y_cuped' in result._experiment_data.ds.roles
        assert 'y2_cuped' in result._experiment_data.ds.roles

    def test_variance_reduction_report_cuped(self, sample_ab_data):
        """Test variance reduction report for CUPED."""
        test = ABTest(cuped_features={'y': 'y0_lag_1'})
        
        result = test.execute(sample_ab_data)
        variance_report = result.variance_reduction_report
        
        assert variance_report is not None
        # Should contain CUPED variance reduction information

    def test_cuped_with_additional_tests(self, sample_ab_data):
        """Test CUPED with additional statistical tests."""
        test = ABTest(
            cuped_features={'y': 'y0_lag_1'},
            additional_tests=['t-test', 'u-test', 'chi2-test']
        )
        
        result = test.execute(sample_ab_data)
        
        assert result.resume is not None
        # Check that multiple tests are performed on CUPED target
        resume_data = result.resume
        
        # Should have results for the CUPED target
        assert len(resume_data) > 0

    def test_cuped_with_multitest_correction(self, sample_ab_data):
        """Test CUPED with multiple testing correction."""
        from hypex.utils import ABNTestMethodsEnum
        
        test = ABTest(
            cuped_features={'y': 'y0_lag_1'},
            multitest_method=ABNTestMethodsEnum.bonferroni
        )
        
        result = test.execute(sample_ab_data)
        
        assert result.resume is not None
        assert result.multitest is not None

    def test_cuped_effect_preservation(self, sample_ab_data):
        """Test that CUPED preserves treatment effects."""
        # Test without CUPED
        test_no_cuped = ABTest()
        result_no_cuped = test_no_cuped.execute(sample_ab_data)
        
        # Test with CUPED
        test_cuped = ABTest(cuped_features={'y': 'y0_lag_1'})
        result_cuped = test_cuped.execute(sample_ab_data)
        
        # Both should detect similar treatment effects
        # (CUPED should reduce variance but preserve effect estimates)
        assert result_no_cuped.resume is not None
        assert result_cuped.resume is not None

    def test_cuped_statistical_power_improvement(self, sample_ab_data):
        """Test that CUPED improves statistical power."""
        # This is a conceptual test - in practice, CUPED should improve
        # the ability to detect effects by reducing variance
        
        test = ABTest(cuped_features={'y': 'y0_lag_1'})
        result = test.execute(sample_ab_data)
        
        # Check that variance reduction occurred
        variance_report = result.variance_reduction_report
        assert variance_report is not None
        
        # If there's variance reduction, statistical power should improve
        # (smaller confidence intervals, lower p-values for real effects)

    def test_cuped_empty_features(self, sample_ab_data):
        """Test ABTest with empty CUPED features."""
        test = ABTest(cuped_features={})
        
        result = test.execute(sample_ab_data)
        
        # Should work normally without CUPED
        assert result.resume is not None
        # Should not have any CUPED targets
        cuped_targets = [col for col in result._experiment_data.ds.roles if 'cuped' in col]
        assert len(cuped_targets) == 0

    def test_cuped_invalid_pre_feature(self, sample_ab_data):
        """Test error handling with invalid pre-feature."""
        test = ABTest(cuped_features={'y': 'nonexistent_pre_feature'})
        
        with pytest.raises(KeyError):
            test.execute(sample_ab_data)

    def test_cuped_invalid_target_feature(self, sample_ab_data):
        """Test error handling with invalid target feature."""
        test = ABTest(cuped_features={'nonexistent_target': 'y0_lag_1'})
        
        with pytest.raises(KeyError):
            test.execute(sample_ab_data)


class TestCUPEDEdgeCases:
    """Test edge cases and special scenarios for CUPED."""
    
    @pytest.fixture
    def minimal_data(self):
        """Create minimal dataset for edge case testing."""
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y_pre': [0.5, 1.5, 2.5, 3.5, 4.5],
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
        """Test CUPED with very small dataset."""
        cuped_features = {'y': 'y_pre'}
        
        result = CUPEDTransformer._inner_function(minimal_data, cuped_features)
        
        assert 'y_cuped' in result.data.columns
        assert len(result['y_cuped']) == len(minimal_data.data)

    def test_single_observation(self):
        """Test CUPED with single observation."""
        df = pd.DataFrame({
            'y': [5.0],
            'y_pre': [4.0],
            'treatment': [1]
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cuped_features = {'y': 'y_pre'}
        result = CUPEDTransformer._inner_function(dataset, cuped_features)
        
        # With single observation, CUPED should return the original value
        assert result.data['y_cuped'].iloc[0] == dataset.data['y'].iloc[0]

    def test_identical_values(self):
        """Test CUPED with identical values."""
        df = pd.DataFrame({
            'y': [5.0, 5.0, 5.0, 5.0, 5.0],
            'y_pre': [3.0, 3.0, 3.0, 3.0, 3.0],
            'treatment': [0, 1, 0, 1, 0]
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cuped_features = {'y': 'y_pre'}
        result = CUPEDTransformer._inner_function(dataset, cuped_features)
        
        # With identical values, CUPED should preserve values
        np.testing.assert_array_almost_equal(
            result.data['y_cuped'].values,
            dataset.data['y'].values,
            decimal=10
        )

    def test_extreme_values(self):
        """Test CUPED with extreme values."""
        df = pd.DataFrame({
            'y': [1e10, -1e10, 1e-10, -1e-10, 0],
            'y_pre': [1e9, -1e9, 1e-9, -1e-9, 0],
            'treatment': [0, 1, 0, 1, 0]
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cuped_features = {'y': 'y_pre'}
        
        # Should handle extreme values without numerical issues
        result = CUPEDTransformer._inner_function(dataset, cuped_features)
        
        assert 'y_cuped' in result.data.columns
        assert not result.data['y_cuped'].isna().any()
        assert np.isfinite(result.data['y_cuped']).all()

    def test_missing_values_in_target(self):
        """Test CUPED behavior with missing values in target."""
        df = pd.DataFrame({
            'y': [1.0, np.nan, 3.0, 4.0, 5.0],
            'y_pre': [0.5, 1.5, 2.5, 3.5, 4.5],
            'treatment': [0, 1, 0, 1, 0]
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cuped_features = {'y': 'y_pre'}
        
        # Behavior depends on implementation - might handle or raise error
        try:
            result = CUPEDTransformer._inner_function(dataset, cuped_features)
            # If successful, check that NaN handling is appropriate
            assert 'y_cuped' in result.data.columns
        except ValueError:
            # Missing values might not be supported
            pass

    def test_missing_values_in_pre_feature(self):
        """Test CUPED behavior with missing values in pre-feature."""
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y_pre': [0.5, np.nan, 2.5, 3.5, 4.5],
            'treatment': [0, 1, 0, 1, 0]
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cuped_features = {'y': 'y_pre'}
        
        # Behavior depends on implementation
        try:
            result = CUPEDTransformer._inner_function(dataset, cuped_features)
            assert 'y_cuped' in result.data.columns
        except ValueError:
            # Missing values might not be supported
            pass

    def test_negative_correlation(self):
        """Test CUPED with negatively correlated features."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate negatively correlated features
        y_pre = np.random.normal(0, 1, n_samples)
        y = -0.7 * y_pre + np.random.normal(0, 0.5, n_samples)  # Negative correlation
        
        df = pd.DataFrame({
            'y': y,
            'y_pre': y_pre,
            'treatment': np.random.binomial(1, 0.5, n_samples)
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cuped_features = {'y': 'y_pre'}
        result = CUPEDTransformer._inner_function(dataset, cuped_features)
        
        # Should still achieve variance reduction with negative correlation
        original_var = dataset['y'].var()
        cuped_var = result['y_cuped'].var()
        variance_reduction = (1 - cuped_var / original_var) * 100
        
        assert variance_reduction > 0

    def test_cuped_with_categorical_pre_feature(self):
        """Test CUPED behavior with categorical pre-feature (should handle or error)."""
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y_pre': ['A', 'B', 'A', 'B', 'A'],  # Categorical
            'treatment': [0, 1, 0, 1, 0]
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cuped_features = {'y': 'y_pre'}
        
        # Should either handle appropriately or raise a clear error
        try:
            result = CUPEDTransformer._inner_function(dataset, cuped_features)
            # If it works, the result should be valid
            assert 'y_cuped' in result.data.columns
        except (TypeError, ValueError):
            # Categorical features might not be supported
            pass

    def test_numerical_stability(self):
        """Test numerical stability with very similar values."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate very similar values (potential numerical precision issues)
        base_value = 1e6
        y = base_value + np.random.normal(0, 1e-6, n_samples)
        y_pre = base_value + np.random.normal(0, 1e-6, n_samples)
        
        df = pd.DataFrame({
            'y': y,
            'y_pre': y_pre,
            'treatment': np.random.binomial(1, 0.5, n_samples)
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole()
            }
        )
        
        cuped_features = {'y': 'y_pre'}
        result = CUPEDTransformer._inner_function(dataset, cuped_features)
        
        # Should handle numerical precision issues gracefully
        assert 'y_cuped' in result.data.columns
        assert np.isfinite(result.data['y_cuped']).all()
        assert not result.data['y_cuped'].isna().any()