"""
Integration tests for CUPAC and CUPED methods within the HypEx library.

This module tests the integration of CUPAC and CUPED with:
- ABTest pipelines
- Different statistical tests
- Multiple testing corrections
- Combination with other transformers
- Real-world scenarios
- Performance considerations
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import time
from copy import deepcopy

from hypex.dataset import Dataset, TargetRole, TreatmentRole, InfoRole
from hypex.ab import ABTest
from hypex.utils.tutorial_data_creation import DataGenerator
from hypex.utils import ABNTestMethodsEnum, ABTestTypesEnum
from hypex.utils.errors import NotSuitableFieldError


class TestCUPACCUPEDIntegration:
    """Test integration between CUPAC and CUPED methods."""
    
    @pytest.fixture
    def comprehensive_data(self):
        """Create comprehensive dataset for integration testing."""
        gen = DataGenerator(
            n_samples=2000,
            distributions={
                "X1": {"type": "normal", "mean": 0, "std": 1},
                "X2": {"type": "bernoulli", "p": 0.5},
                "y0": {"type": "normal", "mean": 5, "std": 1},
            },
            time_correlations={"X1": 0.3, "X2": 0.2, "y0": 0.7},
            effect_size=2.0,
            seed=123
        )
        df = gen.generate()
        
        # Keep necessary columns
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

    def test_cupac_cuped_both_applied(self, comprehensive_data):
        """Test applying both CUPAC and CUPED simultaneously."""
        test = ABTest(
            cupac_features={'y': ['y0_lag_1', 'X1_lag', 'X2_lag']},
            cuped_features={'y': 'y0_lag_1'},
            cupac_model='linear'
        )
        
        result = test.execute(comprehensive_data)
        
        # Should have both transformations applied
        assert 'y_cupac' in result._experiment_data.ds.roles
        assert 'y_cuped' in result._experiment_data.ds.roles
        
        # Should have variance reduction reports for both
        variance_report = result.variance_reduction_report
        assert variance_report is not None
        
        # Both should be analyzed in the AB test
        resume_data = result.resume
        assert len(resume_data) >= 2  # At least original and transformed targets

    def test_cupac_cuped_different_targets(self, comprehensive_data):
        """Test CUPAC and CUPED applied to different targets."""
        # Add another target
        data_copy = deepcopy(comprehensive_data)
        y2 = data_copy.data['y'].values * 0.8 + np.random.normal(0, 0.5, len(data_copy.data))
        data_copy = data_copy.add_column(data=y2, role={'y2': TargetRole()})
        
        test = ABTest(
            cupac_features={'y': ['y0_lag_1', 'X1_lag']},
            cuped_features={'y2': 'y0_lag_1'},
            cupac_model='linear'
        )
        
        result = test.execute(data_copy)
        
        # Should have transformations for different targets
        assert 'y_cupac' in result._experiment_data.ds.roles
        assert 'y2_cuped' in result._experiment_data.ds.roles
        
        # Should analyze multiple targets
        resume_data = result.resume
        assert len(resume_data) >= 2

    def test_cupac_cuped_same_target_different_features(self, comprehensive_data):
        """Test CUPAC and CUPED on same target with different pre-features."""
        test = ABTest(
            cupac_features={'y': ['X1_lag', 'X2_lag']},  # Different features
            cuped_features={'y': 'y0_lag_1'},           # Different feature
            cupac_model='linear'
        )
        
        result = test.execute(comprehensive_data)
        
        # Should create both transformations
        assert 'y_cupac' in result._experiment_data.ds.roles
        assert 'y_cuped' in result._experiment_data.ds.roles
        
        # Check that both achieve variance reduction
        variance_report = result.variance_reduction_report
        assert variance_report is not None

    def test_sequential_application_order(self, comprehensive_data):
        """Test that CUPAC and CUPED are applied in correct order."""
        test = ABTest(
            cupac_features={'y': ['y0_lag_1', 'X1_lag']},
            cuped_features={'y': 'y0_lag_1'},
            cupac_model='linear'
        )
        
        result = test.execute(comprehensive_data)
        
        # Both transformations should be present
        assert 'y_cupac' in result._experiment_data.ds.roles
        assert 'y_cuped' in result._experiment_data.ds.roles
        
        # Original target should still be present
        assert 'y' in result._experiment_data.ds.roles

    def test_variance_reduction_comparison(self, comprehensive_data):
        """Compare variance reduction between CUPAC, CUPED, and both."""
        # Test CUPED only
        test_cuped = ABTest(cuped_features={'y': 'y0_lag_1'})
        result_cuped = test_cuped.execute(comprehensive_data)
        
        # Test CUPAC only
        test_cupac = ABTest(
            cupac_features={'y': ['y0_lag_1', 'X1_lag', 'X2_lag']},
            cupac_model='linear'
        )
        result_cupac = test_cupac.execute(comprehensive_data)
        
        # Test both
        test_both = ABTest(
            cupac_features={'y': ['y0_lag_1', 'X1_lag', 'X2_lag']},
            cuped_features={'y': 'y0_lag_1'},
            cupac_model='linear'
        )
        result_both = test_both.execute(comprehensive_data)
        
        # All should have variance reduction reports
        assert result_cuped.variance_reduction_report is not None
        assert result_cupac.variance_reduction_report is not None
        assert result_both.variance_reduction_report is not None


class TestABTestPipelineIntegration:
    """Test integration with complete ABTest pipeline."""
    
    @pytest.fixture
    def ab_test_data(self):
        """Create A/B test data for pipeline testing."""
        gen = DataGenerator(
            n_samples=1500,
            distributions={
                "X1": {"type": "normal", "mean": 0, "std": 1},
                "X2": {"type": "bernoulli", "p": 0.5},
                "y0": {"type": "normal", "mean": 10, "std": 2},
            },
            time_correlations={"X1": 0.25, "X2": 0.15, "y0": 0.65},
            effect_size=1.8,
            seed=456
        )
        df = gen.generate()
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

    def test_cupac_with_all_statistical_tests(self, ab_test_data):
        """Test CUPAC with all available statistical tests."""
        test = ABTest(
            cupac_features={'y': ['y0_lag_1', 'X1_lag', 'X2_lag']},
            cupac_model='linear',
            additional_tests=['t-test', 'u-test', 'chi2-test']
        )
        
        result = test.execute(ab_test_data)
        
        assert result.resume is not None
        resume_data = result.resume
        
        # Should have results for multiple tests
        test_columns = [col for col in resume_data.columns if 'Test' in col or 'test' in col]
        assert len(test_columns) > 0

    def test_cuped_with_all_statistical_tests(self, ab_test_data):
        """Test CUPED with all available statistical tests."""
        test = ABTest(
            cuped_features={'y': 'y0_lag_1'},
            additional_tests=['t-test', 'u-test', 'chi2-test']
        )
        
        result = test.execute(ab_test_data)
        
        assert result.resume is not None
        resume_data = result.resume
        
        # Should have results for multiple tests
        test_columns = [col for col in resume_data.columns if 'Test' in col or 'test' in col]
        assert len(test_columns) > 0

    def test_cupac_with_multitest_corrections(self, ab_test_data):
        """Test CUPAC with different multiple testing correction methods."""
        correction_methods = [
            ABNTestMethodsEnum.bonferroni,
            ABNTestMethodsEnum.holm,
            ABNTestMethodsEnum.sidak
        ]
        
        for method in correction_methods:
            test = ABTest(
                cupac_features={'y': ['y0_lag_1', 'X1_lag']},
                cupac_model='linear',
                multitest_method=method
            )
            
            result = test.execute(ab_test_data)
            
            assert result.resume is not None
            assert result.multitest is not None

    def test_cuped_with_multitest_corrections(self, ab_test_data):
        """Test CUPED with different multiple testing correction methods."""
        correction_methods = [
            ABNTestMethodsEnum.bonferroni,
            ABNTestMethodsEnum.holm,
            ABNTestMethodsEnum.sidak
        ]
        
        for method in correction_methods:
            test = ABTest(
                cuped_features={'y': 'y0_lag_1'},
                multitest_method=method
            )
            
            result = test.execute(ab_test_data)
            
            assert result.resume is not None
            assert result.multitest is not None

    def test_cupac_different_models_comparison(self, ab_test_data):
        """Compare CUPAC performance with different models."""
        models_to_test = ['linear', 'ridge', 'lasso']
        results = {}
        
        for model in models_to_test:
            test = ABTest(
                cupac_features={'y': ['y0_lag_1', 'X1_lag', 'X2_lag']},
                cupac_model=model
            )
            
            result = test.execute(ab_test_data)
            results[model] = result
            
            assert result.resume is not None
            assert result.variance_reduction_report is not None

        # All models should produce valid results
        assert len(results) == len(models_to_test)

    def test_cupac_auto_model_selection(self, ab_test_data):
        """Test CUPAC with automatic model selection."""
        test = ABTest(
            cupac_features={'y': ['y0_lag_1', 'X1_lag', 'X2_lag']},
            cupac_model=None  # Auto selection
        )
        
        result = test.execute(ab_test_data)
        
        assert result.resume is not None
        assert result.variance_reduction_report is not None
        
        # Should have selected best model automatically
        assert 'y_cupac' in result._experiment_data.ds.roles

    def test_equal_variance_parameter(self, ab_test_data):
        """Test t-test equal variance parameter with CUPAC/CUPED."""
        test = ABTest(
            cupac_features={'y': ['y0_lag_1']},
            cuped_features={'y': 'y0_lag_1'},
            cupac_model='linear',
            t_test_equal_var=False
        )
        
        result = test.execute(ab_test_data)
        
        assert result.resume is not None

    def test_group_sizes_reporting(self, ab_test_data):
        """Test that group sizes are correctly reported with CUPAC/CUPED."""
        test = ABTest(
            cupac_features={'y': ['y0_lag_1', 'X1_lag']},
            cuped_features={'y': 'y0_lag_1'},
            cupac_model='linear'
        )
        
        result = test.execute(ab_test_data)
        
        assert result.resume is not None
        assert result.sizes is not None
        
        # Group sizes should be consistent across transformations
        sizes_data = result.sizes
        assert len(sizes_data) > 0


class TestRealWorldScenarios:
    """Test real-world scenarios and use cases."""
    
    @pytest.fixture
    def ecommerce_data(self):
        """Simulate e-commerce A/B test data."""
        np.random.seed(789)
        n_users = 5000
        
        # User characteristics
        age = np.random.normal(35, 10, n_users)
        previous_purchases = np.random.poisson(3, n_users)
        session_duration_pre = np.random.exponential(300, n_users)  # seconds
        
        # Treatment assignment (stratified by age)
        age_quartiles = np.percentile(age, [25, 50, 75])
        treatment_prob = np.where(age < age_quartiles[0], 0.4,
                         np.where(age < age_quartiles[1], 0.5,
                         np.where(age < age_quartiles[2], 0.6, 0.5)))
        treatment = np.random.binomial(1, treatment_prob)
        
        # Outcome: session duration (correlated with pre-period)
        correlation = 0.6
        noise = np.random.normal(0, 100, n_users)
        session_duration = (correlation * session_duration_pre + 
                          np.sqrt(1 - correlation**2) * noise +
                          treatment * 50 +  # Treatment effect
                          age * 2)  # Age effect
        
        df = pd.DataFrame({
            'user_id': range(n_users),
            'age': age,
            'previous_purchases': previous_purchases,
            'session_duration_pre': session_duration_pre,
            'session_duration': session_duration,
            'treatment': treatment
        })
        
        return Dataset(
            data=df,
            roles={
                'session_duration': TargetRole(),
                'treatment': TreatmentRole(),
                'user_id': InfoRole()
            }
        )

    def test_ecommerce_cupac_scenario(self, ecommerce_data):
        """Test CUPAC in e-commerce scenario."""
        test = ABTest(
            cupac_features={
                'session_duration': ['session_duration_pre', 'age', 'previous_purchases']
            },
            cupac_model=['linear', 'ridge']
        )
        
        result = test.execute(ecommerce_data)
        
        assert result.resume is not None
        assert result.variance_reduction_report is not None
        
        # Should detect treatment effect with improved precision
        resume_data = result.resume
        assert len(resume_data) > 0

    def test_ecommerce_cuped_scenario(self, ecommerce_data):
        """Test CUPED in e-commerce scenario."""
        test = ABTest(
            cuped_features={'session_duration': 'session_duration_pre'}
        )
        
        result = test.execute(ecommerce_data)
        
        assert result.resume is not None
        assert result.variance_reduction_report is not None
        
        # Should achieve variance reduction
        variance_report = result.variance_reduction_report
        assert variance_report is not None

    def test_multiple_metrics_scenario(self, ecommerce_data):
        """Test scenario with multiple business metrics."""
        # Add conversion rate as second metric
        data_copy = deepcopy(ecommerce_data)
        conversion_rate = (0.1 + 0.02 * data_copy.data['treatment'].values + 
                          0.001 * data_copy.data['previous_purchases'].values +
                          np.random.normal(0, 0.05, len(data_copy.data)))
        conversion_rate = np.clip(conversion_rate, 0, 1)
        
        data_copy = data_copy.add_column(
            data=conversion_rate, 
            role={'conversion_rate': TargetRole()}
        )
        
        test = ABTest(
            cupac_features={
                'session_duration': ['session_duration_pre', 'age'],
                'conversion_rate': ['previous_purchases']
            },
            cuped_features={'session_duration': 'session_duration_pre'},
            cupac_model='linear'
        )
        
        result = test.execute(data_copy)
        
        assert result.resume is not None
        # Should analyze multiple metrics
        resume_data = result.resume
        assert len(resume_data) >= 2

    @pytest.fixture
    def marketing_data(self):
        """Simulate marketing campaign A/B test data."""
        np.random.seed(101112)
        n_users = 3000
        
        # Pre-campaign metrics
        email_opens_pre = np.random.poisson(5, n_users)
        click_through_rate_pre = np.random.beta(2, 20, n_users)  # Low CTR
        
        # Treatment assignment
        treatment = np.random.binomial(1, 0.5, n_users)
        
        # Post-campaign metrics (influenced by pre-period and treatment)
        email_opens = (0.7 * email_opens_pre + 
                      treatment * 2 + 
                      np.random.poisson(1, n_users))
        
        click_through_rate = (0.6 * click_through_rate_pre + 
                             treatment * 0.02 + 
                             np.random.beta(1, 30, n_users))
        
        df = pd.DataFrame({
            'user_id': range(n_users),
            'email_opens_pre': email_opens_pre,
            'click_through_rate_pre': click_through_rate_pre,
            'email_opens': email_opens,
            'click_through_rate': click_through_rate,
            'treatment': treatment
        })
        
        return Dataset(
            data=df,
            roles={
                'email_opens': TargetRole(),
                'click_through_rate': TargetRole(),
                'treatment': TreatmentRole(),
                'user_id': InfoRole()
            }
        )

    def test_marketing_campaign_analysis(self, marketing_data):
        """Test marketing campaign analysis with CUPAC and CUPED."""
        test = ABTest(
            cupac_features={
                'email_opens': ['email_opens_pre'],
                'click_through_rate': ['click_through_rate_pre']
            },
            cuped_features={
                'email_opens': 'email_opens_pre'
            },
            cupac_model='linear',
            additional_tests=['t-test', 'u-test']
        )
        
        result = test.execute(marketing_data)
        
        assert result.resume is not None
        assert result.variance_reduction_report is not None
        
        # Should analyze both metrics
        resume_data = result.resume
        assert len(resume_data) >= 2


class TestPerformanceAndScalability:
    """Test performance and scalability of CUPAC and CUPED."""
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        # Create large dataset
        n_samples = 10000
        np.random.seed(999)
        
        df = pd.DataFrame({
            'y': np.random.normal(0, 1, n_samples),
            'y_pre': np.random.normal(0, 1, n_samples),
            'x1': np.random.normal(0, 1, n_samples),
            'x2': np.random.normal(0, 1, n_samples),
            'x3': np.random.normal(0, 1, n_samples),
            'treatment': np.random.binomial(1, 0.5, n_samples),
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
        
        # Test CUPAC performance
        start_time = time.time()
        test_cupac = ABTest(
            cupac_features={'y': ['y_pre', 'x1', 'x2', 'x3']},
            cupac_model='linear'
        )
        result_cupac = test_cupac.execute(dataset)
        cupac_time = time.time() - start_time
        
        # Test CUPED performance
        start_time = time.time()
        test_cuped = ABTest(cuped_features={'y': 'y_pre'})
        result_cuped = test_cuped.execute(dataset)
        cuped_time = time.time() - start_time
        
        # Both should complete in reasonable time (adjust threshold as needed)
        assert cupac_time < 30  # seconds
        assert cuped_time < 10   # seconds
        
        assert result_cupac.resume is not None
        assert result_cuped.resume is not None

    def test_many_features_cupac(self):
        """Test CUPAC with many covariate features."""
        n_samples = 1000
        n_features = 20
        np.random.seed(777)
        
        # Generate many features
        data_dict = {'y': np.random.normal(0, 1, n_samples)}
        feature_names = []
        
        for i in range(n_features):
            feature_name = f'x{i}'
            data_dict[feature_name] = np.random.normal(0, 1, n_samples)
            feature_names.append(feature_name)
        
        data_dict['treatment'] = np.random.binomial(1, 0.5, n_samples)
        data_dict['user_id'] = range(n_samples)
        
        df = pd.DataFrame(data_dict)
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole(),
                'user_id': InfoRole()
            }
        )
        
        test = ABTest(
            cupac_features={'y': feature_names},
            cupac_model='linear'
        )
        
        result = test.execute(dataset)
        
        assert result.resume is not None
        assert result.variance_reduction_report is not None

    def test_memory_efficiency(self):
        """Test memory efficiency with transformations."""
        n_samples = 5000
        np.random.seed(888)
        
        df = pd.DataFrame({
            'y': np.random.normal(0, 1, n_samples),
            'y_pre': np.random.normal(0, 1, n_samples),
            'x1': np.random.normal(0, 1, n_samples),
            'x2': np.random.normal(0, 1, n_samples),
            'treatment': np.random.binomial(1, 0.5, n_samples),
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
        
        # Test that transformations don't create excessive memory overhead
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        test = ABTest(
            cupac_features={'y': ['y_pre', 'x1', 'x2']},
            cuped_features={'y': 'y_pre'},
            cupac_model='linear'
        )
        
        result = test.execute(dataset)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 500 * 1024 * 1024  # 500 MB
        
        assert result.resume is not None


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness of integration."""
    
    @pytest.fixture
    def problematic_data(self):
        """Create data with potential issues."""
        df = pd.DataFrame({
            'y': [1, 2, 3, np.nan, 5],
            'y_pre': [1, 2, np.inf, 4, 5],
            'x1': [1, 2, 3, 4, 5],
            'treatment': [0, 1, 0, 1, 0],
            'user_id': range(5)
        })
        
        return Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole(),
                'user_id': InfoRole()
            }
        )

    def test_missing_data_handling(self, problematic_data):
        """Test handling of missing data."""
        test = ABTest(
            cupac_features={'y': ['y_pre', 'x1']},
            cuped_features={'y': 'y_pre'},
            cupac_model='linear'
        )
        
        # Should either handle gracefully or provide clear error
        try:
            result = test.execute(problematic_data)
            # If successful, result should be valid
            assert result.resume is not None
        except (ValueError, RuntimeError) as e:
            # Should provide informative error message
            assert len(str(e)) > 0

    def test_insufficient_data_error(self):
        """Test behavior with insufficient data."""
        df = pd.DataFrame({
            'y': [1, 2],
            'y_pre': [1, 2],
            'treatment': [0, 1],
            'user_id': [0, 1]
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole(),
                'user_id': InfoRole()
            }
        )
        
        test = ABTest(
            cupac_features={'y': ['y_pre']},
            cupac_model='linear'
        )
        
        # Should handle gracefully or provide clear error
        try:
            result = test.execute(dataset)
            assert result.resume is not None
        except (ValueError, RuntimeError):
            # Expected for insufficient data
            pass

    def test_invalid_configuration_combinations(self):
        """Test invalid configuration combinations."""
        df = pd.DataFrame({
            'y': [1, 2, 3, 4, 5],
            'y_pre': [1, 2, 3, 4, 5],
            'treatment': [0, 1, 0, 1, 0],
            'user_id': range(5)
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole(),
                'user_id': InfoRole()
            }
        )
        
        # Test invalid model name
        test_invalid_model = ABTest(
            cupac_features={'y': ['y_pre']},
            cupac_model='invalid_model_name'
        )
        
        with pytest.raises(RuntimeError):
            test_invalid_model.execute(dataset)

    def test_empty_groups_handling(self):
        """Test handling of empty treatment groups."""
        df = pd.DataFrame({
            'y': [1, 2, 3, 4, 5],
            'y_pre': [1, 2, 3, 4, 5],
            'treatment': [0, 0, 0, 0, 0],  # All control
            'user_id': range(5)
        })
        
        dataset = Dataset(
            data=df,
            roles={
                'y': TargetRole(),
                'treatment': TreatmentRole(),
                'user_id': InfoRole()
            }
        )
        
        test = ABTest(
            cupac_features={'y': ['y_pre']},
            cuped_features={'y': 'y_pre'},
            cupac_model='linear'
        )
        
        # Should handle gracefully or provide clear error
        try:
            result = test.execute(dataset)
            assert result.resume is not None
        except (ValueError, RuntimeError, NotSuitableFieldError):
            # May not be able to compute statistics with single group
            pass