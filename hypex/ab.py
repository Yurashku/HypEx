from __future__ import annotations

from typing import Literal

from .analyzers.ab import ABAnalyzer
from .comparators import Chi2Test, GroupDifference, GroupSizes, TTest, UTest
from .dataset import TargetRole, TreatmentRole
from .experiments.base import Experiment, OnRoleExperiment
from .ui.ab import ABOutput
from .ui.base import ExperimentShell
from .utils import ABNTestMethodsEnum
from .transformers import CUPEDTransformer


class ABTest(ExperimentShell):
    """A class for conducting A/B tests with configurable statistical tests and multiple testing correction.

    This class provides functionality to run A/B tests with options for different statistical tests
    (t-test, u-test, chi-square test) and multiple testing correction methods.

    Args:
        additional_tests (Union[str, List[str], None], optional): Statistical test(s) to run in addition to
            the default group difference calculation. Valid options are "t-test", "u-test", and "chi2-test".
            Can be a single test name or list of test names. Defaults to ["t-test"].
        multitest_method (str, optional): Method to use for multiple testing correction. Valid options are:
            "bonferroni", "sidak", "holm-sidak", "holm", "simes-hochberg", "hommel", "fdr_bh", "fdr_by",
            "fdr_tsbh", "fdr_tsbhy", "quantile". Defaults to "holm".

            For more information refer to the statsmodels documentation:
            https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html

    Examples
    --------
    .. code-block:: python

        # Basic A/B test with default t-test
        ab_test = ABTest()
        results = ab_test.execute(data)

        # A/B test with multiple statistical tests
        ab_test = ABTest(
            additional_tests=["t-test", "chi2-test"],
            multitest_method="bonferroni",
            cuped_features={"target_feature": "pre_target_feature"}
        )
        results = ab_test.execute(data)
    """

    @staticmethod
    def _make_experiment(additional_tests, multitest_method, cuped_features=None, cupac_features=None, cupac_model=None):
        test_mapping = {
            "t-test": TTest(compare_by="groups", grouping_role=TreatmentRole()),
            "u-test": UTest(compare_by="groups", grouping_role=TreatmentRole()),
            "chi2-test": Chi2Test(compare_by="groups", grouping_role=TreatmentRole()),
        }
        on_role_executors = [GroupDifference(grouping_role=TreatmentRole())]
        additional_tests = ["t-test"] if additional_tests is None else additional_tests
        additional_tests = (
            additional_tests
            if isinstance(additional_tests, list)
            else [additional_tests]
        )
        for i in additional_tests:
            on_role_executors += [test_mapping[i]]



        if cuped_features and cupac_features:
            raise ValueError("You can use only one transformer/executor: either CUPED or CUPACExecutor, not both.")

        # Build base executors list
        executors = [
            GroupSizes(grouping_role=TreatmentRole()),
            OnRoleExperiment(
                executors=on_role_executors,
                role=TargetRole(),
            ),
            ABAnalyzer(
                multitest_method=(
                    ABNTestMethodsEnum(multitest_method)
                    if multitest_method
                    else None
                )
            ),
        ]
        if cuped_features:
            executors.insert(0, CUPEDTransformer(cuped_features=cuped_features))
        elif cupac_features:
            from .ml import CUPACExecutor
            executors.insert(0, CUPACExecutor(cupac_features=cupac_features, cupac_model=cupac_model))

        return Experiment(executors=executors)

    def __init__(
        self,
        additional_tests: (
            Literal["t-test", "u-test", "chi2-test"]
            | list[Literal["t-test", "u-test", "chi2-test"]]
            | None
        ) = None,
        multitest_method: (
            Literal[
                "bonferroni",
                "sidak",
                "holm-sidak",
                "holm",
                "simes-hochberg",
                "hommel",
                "fdr_bh",
                "fdr_by",
                "fdr_tsbh",
                "fdr_tsbhy",
                "quantile",
            ]
            | None
        ) = "holm",
        t_test_equal_var: bool | None = None,
        cuped_features: dict[str, str] | None = None,
        cupac_features: dict | None = None,
        cupac_model: str | list[str] | None = None,
        ):
        """
        Args:
            additional_tests: Statistical test(s) to run in addition to the default group difference calculation. Valid options are "t-test", "u-test", and "chi2-test". Can be a single test name or list of test names. Defaults to ["t-test"].
            multitest_method: Method to use for multiple testing correction. Valid options are: "bonferroni", "sidak", "holm-sidak", "holm", "simes-hochberg", "hommel", "fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbhy", "quantile". Defaults to "holm".
            t_test_equal_var: Whether to use equal variance in t-test (optional).
            cuped_features: dict[str, str] — Dictionary {target_feature: pre_target_feature} for CUPED. Only dict is allowed.
            cupac_features: dict — Parameters for CUPAC, e.g. {"target1": ["cov1", "cov2"], ...}.
            cupac_model: str or list of str — model name (e.g. 'linear', 'ridge', 'lasso', 'catboost') or list of model names to try. If None, all available models will be tried and the best will be selected by variance reduction.
        Raises:
            ValueError: If both cuped_features and cupac_features are specified.
        """
        super().__init__(
            experiment=self._make_experiment(additional_tests, multitest_method, cuped_features, cupac_features, cupac_model),
            output=ABOutput(),
        )
        if t_test_equal_var is not None:
            self.experiment.set_params({TTest: {"calc_kwargs": {"equal_var": t_test_equal_var}}})
