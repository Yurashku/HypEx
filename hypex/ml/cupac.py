# --- Новый CUPACML ---
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from ..executor.executor import MLExecutor
from ..utils.adapter import Adapter
from ..dataset.dataset import Dataset, ExperimentData
from ..dataset.roles import FeatureRole, TargetRole

class CUPACML(MLExecutor):
    def __init__(
        self,
        cupac_features: dict[str, list[str]],
        models: Optional[Dict[str, Any]] = None,
        n_folds: int = 5,
        random_state: Optional[int] = None,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.cupac_features = cupac_features
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = models or {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=0.5),
            "Lasso": Lasso(alpha=0.01, max_iter=10000),
            "CatBoost": CatBoostRegressor(
                iterations=100,
                depth=4,
                learning_rate=0.1,
                silent=True,
                random_state=random_state,
                allow_writing_files=False,
            ),
        }
        self.results_ = {}

    def execute(self, data: ExperimentData) -> ExperimentData:
        df = Adapter.to_pandas(data.ds)
        result_df = df.copy()
        for target, covariates in self.cupac_features.items():
            # временно назначаем роли
            tmp_roles = {col: FeatureRole() for col in covariates}
            tmp_roles[target] = TargetRole()
            tmp_dataset = Dataset(data=df, roles=tmp_roles)
            self.fit(tmp_dataset)
            y = df[target]
            X_inf = df[covariates]
            model = self.models_fitted_[target]["model"]
            pred = model.predict(X_inf)
            pred_centered = pred - pred.mean()
            theta = np.cov(y, pred_centered)[0, 1] / pred_centered.var()
            y_adj = y - theta * pred_centered
            result_df[f"{target}_cupac"] = y_adj
        return data.copy(data=Adapter.to_dataset(result_df))

    def _prepare_train_data(self, df: pd.DataFrame, target: str, covariates: list[str]):
        X = df[covariates]
        y = df[target]
        return X, y

    def _prepare_inference_data(self, df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
        missing = [col for col in covariates if col not in df.columns]
        if missing:
            raise ValueError(f"Missing covariates: {missing}")
        return df[covariates]

    def fit(self, X: 'Dataset', Y: Optional['Dataset'] = None) -> 'CUPACML':
        df = Adapter.to_pandas(X)
        self.models_fitted_ = {}
        for target, covariates in self.cupac_features.items():
            X_train, y_train = self._prepare_train_data(df, target, covariates)
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            results = {}
            for name, model in self.models.items():
                fold_scores = []
                fold_var_reductions = []
                status = "success"
                try:
                    for train_idx, val_idx in kf.split(X_train):
                        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                        if name == "CatBoost":
                            m = CatBoostRegressor(**model.get_params())
                            m.fit(X_tr, y_tr, verbose=False)
                        else:
                            m = model.__class__(**model.get_params())
                            m.fit(X_tr, y_tr)
                        pred = m.predict(X_val)
                        fold_scores.append(r2_score(y_val, pred))
                        fold_var_reductions.append(self._calculate_variance_reduction(y_val, pred))
                    results[name] = {
                        "r2": np.nanmean(fold_scores),
                        "var_reduction": np.nanmean(fold_var_reductions),
                        "status": status,
                    }
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    results[name] = {
                        "r2": None,
                        "var_reduction": None,
                        "status": f"failed: {error_msg}",
                    }
            successful_models = {k: v for k, v in results.items() if v["status"] == "success"}
            if not successful_models:
                raise RuntimeError(f"Все модели завершились с ошибкой для {target}")
            best_model_name = max(successful_models, key=lambda x: successful_models[x]["var_reduction"])
            best_model_params = self.models[best_model_name].get_params()
            if best_model_name == "CatBoost":
                best_model = CatBoostRegressor(**best_model_params)
                best_model.fit(X_train, y_train, verbose=False)
                feature_importances = dict(zip(X_train.columns, best_model.get_feature_importance()))
            else:
                best_model = self.models[best_model_name].__class__(**best_model_params)
                best_model.fit(X_train, y_train)
                if hasattr(best_model, "coef_"):
                    feature_importances = dict(zip(X_train.columns, best_model.coef_))
                else:
                    feature_importances = None
            self.models_fitted_[target] = {
                "model": best_model,
                "model_name": best_model_name,
                "feature_importances": feature_importances,
                "results": results,
            }
        self.is_fitted = True
        return self

    def predict(self, X: 'Dataset') -> 'Dataset':
        # Преобразуем к pandas.DataFrame
        df = Adapter.to_pandas(X)
        X_inf = self._prepare_inference_data(df)
        pred = self.best_model.predict(X_inf)
        # Возвращаем Dataset с предсказанными значениями
        return Adapter.to_dataset(pred, roles={f"{self.target_col}_cupac": X.roles.get(self.target_col)})

    def transform(self, X: 'Dataset') -> 'Dataset':
        df = Adapter.to_pandas(X)
        for target, covariates in self.cupac_features.items():
            y = df[target]
            X_inf = self._prepare_inference_data(df, covariates)
            model = self.models_fitted_[target]["model"]
            pred = model.predict(X_inf)
            pred_centered = pred - pred.mean()
            theta = np.cov(y, pred_centered)[0, 1] / pred_centered.var()
            y_adj = y - theta * pred_centered
            df[f"{target}_cupac"] = y_adj
        return Adapter.to_dataset(df)

    @classmethod
    def _inner_function(
        cls,
        data: 'Dataset',
        **kwargs,
    ) -> 'Dataset':
        # Для совместимости с pipeline: обучаем и трансформируем
        self = kwargs.get('self')
        if not self.is_fitted:
            self.fit(data)
        return self.transform(data)

    def execute(self, data: 'ExperimentData') -> 'ExperimentData':
        # Интеграция с ExperimentData
        result = data.copy(
            data=self.calc(
                data=data.ds,
                self=self,
            )
        )
        return result

    def _calculate_variance_reduction(self, y: pd.Series, pred: pd.Series) -> float:
        pred_centered = pred - pred.mean()
        if pred_centered.var() < 1e-10:
            return 0.0
        theta = np.cov(y, pred_centered)[0, 1] / pred_centered.var()
        y_adj = y - theta * pred_centered
        return max(0, (1 - y_adj.var() / y.var()) * 100)

    def get_report(self) -> str:
        if not self.is_fitted:
            return "Модель не обучена. Сначала вызовите fit()."
        sorted_features = (
            sorted(self.feature_importances_.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            if self.feature_importances_ else []
        )
        model_comparison = []
        for name, data in self.model_results_.items():
            if data["status"] != "success":
                line = f"{name}: {data['status']}"
            else:
                line = f"{name}: R²={data['r2']:.3f}, Var.Red.={data['var_reduction']:.1f}%"
            model_comparison.append(line)
        feature_analysis = []
        if sorted_features:
            max_coef = max(abs(v) for _, v in sorted_features)
            for feat, coef in sorted_features:
                rel_impact = abs(coef) / max_coef if max_coef != 0 else 0
                feature_analysis.append(f"- {feat:<25} {coef:>7.3f} {'▇'*int(10*rel_impact)}")
        report = [
            "Расширенный CUPAC Report",
            "=" * 40,
            "Сравнение моделей:",
            *model_comparison,
            "",
            f"Лучшая модель: {self.best_model_name}",
            f"Снижение дисперсии: {self.variance_reduction:.1f}%",
            f"Качество предсказания (R²): {self.best_score:.3f}",
            "",
            "Топ-10 значимых признаков:",
            *(feature_analysis if feature_analysis else ["Нет данных о важности признаков"]),
            "",
            "Интерпретация:",
            "▇▇▇▇▇▇▇▇▇▇ - максимальное влияние",
            "Коэффициенты > 0: положительная связь с целевой переменной",
            "Коэффициенты < 0: отрицательная связь",
        ]
        return "\n".join(report)
