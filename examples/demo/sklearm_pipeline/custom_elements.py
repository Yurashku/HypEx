import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import warnings


# =============================================================================
# FEATURE ENGINEERING COMPONENTS
# =============================================================================

class RatioFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Генератор признаков-соотношений (x_i / x_j).
    Создает новые признаки как отношения исходных признаков.
    """
    
    def __init__(self, max_features: int = 10, min_std_threshold: float = 0.01):
        """
        max_features: максимальное количество соотношений для создания
        min_std_threshold: минимальное стандартное отклонение для включения признака
        """
        self.max_features = max_features
        self.min_std_threshold = min_std_threshold
        self.selected_pairs_ = None
        self.feature_names_ = None
        
    def fit(self, X, target=None):
        """Выбираем наиболее информативные пары для соотношений"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            self.feature_names_ = X.columns.tolist()
        else:
            X_array = X
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Находим признаки с достаточной вариабельностью
        valid_features = []
        for i in range(X_array.shape[1]):
            if np.std(X_array[:, i]) > self.min_std_threshold:
                valid_features.append(i)
        
        # Создаем пары для соотношений
        pairs = []
        for i in valid_features:
            for j in valid_features:
                if i != j:
                    # Проверяем, что знаменатель не содержит много нулей
                    denominator = X_array[:, j]
                    zero_ratio = np.sum(np.abs(denominator) < 1e-8) / len(denominator)
                    if zero_ratio < 0.1:  # Менее 10% нулевых значений
                        pairs.append((i, j))
        
        # Ограничиваем количество пар
        if len(pairs) > self.max_features:
            # Выбираем случайно, но с фиксированным seed для воспроизводимости
            np.random.seed(42)
            indices = np.random.choice(len(pairs), self.max_features, replace=False)
            pairs = [pairs[i] for i in indices]
        else:
            pairs = pairs[:self.max_features]
            
        self.selected_pairs_ = pairs
        return self
    
    def transform(self, X):
        """Создаем признаки-соотношения"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        ratio_features = []
        
        for i, j in self.selected_pairs_:
            numerator = X_array[:, i]
            denominator = X_array[:, j]
            
            # Защита от деления на ноль
            denominator = np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)
            ratio = numerator / denominator
            
            # Обрезаем экстремальные значения
            ratio = np.clip(ratio, -1e6, 1e6)
            ratio_features.append(ratio.reshape(-1, 1))
        
        if ratio_features:
            return np.hstack(ratio_features)
        else:
            # Возвращаем пустой массив правильной формы
            return np.empty((X_array.shape[0], 0))


class LogFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Генератор логарифмических признаков.
    Применяет log1p преобразование к положительным признакам.
    """
    
    def __init__(self, min_positive_ratio: float = 0.8):
        """
        min_positive_ratio: минимальная доля положительных значений для применения log
        """
        self.min_positive_ratio = min_positive_ratio
        self.log_features_mask_ = None
        self.feature_names_ = None
        
    def fit(self, X, target=None):
        """Определяем признаки подходящие для логарифмирования"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            self.feature_names_ = X.columns.tolist()
        else:
            X_array = X
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.log_features_mask_ = []
        
        for i in range(X_array.shape[1]):
            col_data = X_array[:, i]
            positive_ratio = np.sum(col_data > 0) / len(col_data)
            
            # Проверяем долю положительных значений и вариабельность
            if (positive_ratio >= self.min_positive_ratio and 
                np.std(col_data) > 0.01):
                self.log_features_mask_.append(i)
        
        return self
    
    def transform(self, X):
        """Создаем логарифмические признаки"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        log_features = []
        
        for i in self.log_features_mask_:
            col_data = X_array[:, i]
            # Используем log1p для стабильности и обработки нулей
            log_feature = np.log1p(np.maximum(col_data, 0))
            log_features.append(log_feature.reshape(-1, 1))
        
        if log_features:
            return np.hstack(log_features)
        else:
            return np.empty((X_array.shape[0], 0))


# =============================================================================
# OUTLIER HANDLING COMPONENTS  
# =============================================================================

class IQROutlierHandler(BaseEstimator, TransformerMixin):
    """
    Обработчик выбросов методом межквартильного размаха (IQR).
    """
    
    def __init__(self, factor: float = 1.5):
        """
        factor: коэффициент для расчета границ (обычно 1.5 или 3.0)
        """
        self.factor = factor
        self.bounds_ = None
        
    def fit(self, X, target=None):
        """Вычисляем IQR границы для каждого признака"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        self.bounds_ = {}
        
        for i in range(X_array.shape[1]):
            col_data = X_array[:, i]
            Q1 = np.percentile(col_data, 25)
            Q3 = np.percentile(col_data, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR
            
            self.bounds_[i] = (lower_bound, upper_bound)
        
        return self
    
    def transform(self, X):
        """Обрезаем выбросы до IQR границ"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values.copy()
        else:
            X_array = X.copy()
            
        for i, (lower_bound, upper_bound) in self.bounds_.items():
            X_array[:, i] = np.clip(X_array[:, i], lower_bound, upper_bound)
            
        return X_array


class ZScoreOutlierHandler(BaseEstimator, TransformerMixin):
    """
    Обработчик выбросов методом Z-score.
    """
    
    def __init__(self, threshold: float = 3.0):
        """
        threshold: количество стандартных отклонений для определения выброса
        """
        self.threshold = threshold
        self.bounds_ = None
        
    def fit(self, X, target=None):
        """Вычисляем границы по Z-score"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        self.bounds_ = {}
        
        for i in range(X_array.shape[1]):
            col_data = X_array[:, i]
            mean_val = np.mean(col_data)
            std_val = np.std(col_data)
            
            lower_bound = mean_val - self.threshold * std_val
            upper_bound = mean_val + self.threshold * std_val
            
            self.bounds_[i] = (lower_bound, upper_bound)
        
        return self
    
    def transform(self, X):
        """Обрезаем выбросы до Z-score границ"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values.copy()
        else:
            X_array = X.copy()
            
        for i, (lower_bound, upper_bound) in self.bounds_.items():
            X_array[:, i] = np.clip(X_array[:, i], lower_bound, upper_bound)
            
        return X_array


class PercentileOutlierHandler(BaseEstimator, TransformerMixin):
    """
    Обработчик выбросов методом процентилей.
    """
    
    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0):
        """
        lower_percentile: нижний процентиль для обрезки
        upper_percentile: верхний процентиль для обрезки
        """
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.bounds_ = None
        
    def fit(self, X, target=None):
        """Вычисляем границы по процентилям"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        self.bounds_ = {}
        
        for i in range(X_array.shape[1]):
            col_data = X_array[:, i]
            lower_bound = np.percentile(col_data, self.lower_percentile)
            upper_bound = np.percentile(col_data, self.upper_percentile)
            
            self.bounds_[i] = (lower_bound, upper_bound)
        
        return self
    
    def transform(self, X):
        """Обрезаем выбросы до процентильных границ"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values.copy()
        else:
            X_array = X.copy()
            
        for i, (lower_bound, upper_bound) in self.bounds_.items():
            X_array[:, i] = np.clip(X_array[:, i], lower_bound, upper_bound)
            
        return X_array


# =============================================================================
# RESULT SAVING COMPONENT
# =============================================================================

class ResultSaver:
    """
    Универсальный сохранятель результатов экспериментов в JSON.
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self._ensure_results_dir()
        
    def _ensure_results_dir(self):
        """Создаем директорию для результатов если не существует"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def calculate_metrics(self, target_true: np.ndarray, target_pred: np.ndarray) -> Dict[str, float]:
        """Вычисление стандартных метрик регрессии"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            metrics = {
                'mape': float(mean_absolute_percentage_error(target_true, target_pred)),
                'r2': float(r2_score(target_true, target_pred)),
                'mae': float(np.mean(np.abs(target_true - target_pred))),
                'rmse': float(np.sqrt(np.mean((target_true - target_pred) ** 2))),
                'mean_target': float(np.mean(target_true)),
                'std_target': float(np.std(target_true)),
                'mean_pred': float(np.mean(target_pred)),
                'std_pred': float(np.std(target_pred))
            }
            
        return metrics
    
    def save_experiment(
        self,
        experiment_name: str,
        model_info: Dict[str, Any],
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        data_info: Dict[str, Any],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Сохранение полной информации об эксперименте
        
        Returns:
            Путь к сохраненному файлу
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        experiment_data = {
            'experiment_info': {
                'name': experiment_name,
                'timestamp': timestamp,
                'datetime': datetime.now().isoformat()
            },
            'model_info': model_info,
            'data_info': data_info,
            'metrics': {
                'train': train_metrics,
                'test': test_metrics,
                'performance': {
                    'train_test_r2_diff': train_metrics['r2'] - test_metrics['r2'],
                    'train_test_mape_diff': test_metrics['mape'] - train_metrics['mape']
                }
            }
        }
        
        if additional_info:
            experiment_data['additional_info'] = additional_info
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)
            
        print(f"Результаты эксперимента сохранены в: {filepath}")
        return filepath
    
    def load_experiment(self, filepath: str) -> Dict[str, Any]:
        """Загрузка результатов эксперимента"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_latest_experiment(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Получение последнего эксперимента с заданным именем"""
        if not os.path.exists(self.results_dir):
            return None
            
        files = [f for f in os.listdir(self.results_dir) 
                if f.startswith(experiment_name) and f.endswith('.json')]
        
        if not files:
            return None
            
        latest_file = max(files)
        return self.load_experiment(os.path.join(self.results_dir, latest_file))