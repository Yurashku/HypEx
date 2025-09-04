import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor
import joblib
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from custom_elements import (
    RatioFeatureGenerator, 
    LogFeatureGenerator,
    IQROutlierHandler, 
    ZScoreOutlierHandler, 
    PercentileOutlierHandler,
    ResultSaver
)


def load_california_housing():
    """
    Загрузка датасета California Housing из sklearn.
    
    Returns:
        tuple: (features_df, target_series, feature_names, description)
    """
    print("📊 Загрузка датасета California Housing...")
    
    # Загружаем данные
    housing = fetch_california_housing(as_frame=True)
    
    # Создаем DataFrame
    features_df = housing.data
    target_series = housing.target
    feature_names = housing.feature_names
    
    print(f"✅ Данные загружены:")
    print(f"   • Количество объектов: {features_df.shape[0]:,}")
    print(f"   • Количество признаков: {features_df.shape[1]}")
    print(f"   • Признаки: {feature_names}")
    print(f"   • Целевая переменная: стоимость жилья (сотни тысяч долларов)")
    print(f"   • Диапазон target: {target_series.min():.2f} - {target_series.max():.2f}")
    
    return features_df, target_series, feature_names, housing.DESCR


def create_feature_engineering_pipeline():
    """
    Создание pipeline для feature engineering с использованием 
    специализированных компонентов и sklearn.
    """
    print("\n🎨 Создание Feature Engineering Pipeline...")
    
    # Создаем FeatureUnion для параллельного применения разных типов признаков
    feature_engineering = FeatureUnion([
        # 1. Исходные признаки (проходят без изменений)
        ('original', 'passthrough'),
        
        # 2. Полиномиальные признаки (sklearn)
        ('polynomial', PolynomialFeatures(
            degree=2, 
            include_bias=False, 
            interaction_only=False
        )),
        
        # 3. Только взаимодействия признаков (sklearn)  
        ('interactions', PolynomialFeatures(
            degree=2,
            include_bias=False,
            interaction_only=True
        )),
        
        # 4. Соотношения признаков (кастомный)
        ('ratios', RatioFeatureGenerator(
            max_features=8,
            min_std_threshold=0.01
        )),
        
        # 5. Логарифмические признаки (кастомный)
        ('logarithmic', LogFeatureGenerator(
            min_positive_ratio=0.8
        )),
        
        # 6. Биннинг признаков (sklearn)
        ('binning', KBinsDiscretizer(
            n_bins=5,
            encode='ordinal',
            strategy='quantile'
        ))
    ])
    
    print("✅ Feature Engineering создан с компонентами:")
    for name, transformer in feature_engineering.transformer_list:
        if hasattr(transformer, '__class__'):
            print(f"   • {name}: {transformer.__class__.__name__}")
        else:
            print(f"   • {name}: {transformer}")
    
    return feature_engineering


def create_ml_pipeline(outlier_method='iqr', random_state=42):
    """
    Создание полного ML pipeline с выбираемым методом обработки выбросов.
    
    Args:
        outlier_method: 'iqr', 'zscore', или 'percentile'
    """
    print(f"\n🔧 Создание ML Pipeline с {outlier_method.upper()} outlier handler...")
    
    # Выбор обработчика выбросов
    if outlier_method == 'iqr':
        outlier_handler = IQROutlierHandler(factor=1.5)
    elif outlier_method == 'zscore':
        outlier_handler = ZScoreOutlierHandler(threshold=3.0)
    elif outlier_method == 'percentile':
        outlier_handler = PercentileOutlierHandler(
            lower_percentile=1.0, 
            upper_percentile=99.0
        )
    else:
        raise ValueError(f"Unknown outlier method: {outlier_method}")
    
    # Создание feature engineering pipeline
    feature_engineering = create_feature_engineering_pipeline()
    
    # Полный pipeline
    pipeline = Pipeline([
        # 1. Обработка выбросов
        ('outlier_handler', outlier_handler),
        
        # 2. Генерация признаков (параллельно)
        ('feature_engineering', feature_engineering),
        
        # 3. Масштабирование признаков
        ('scaler', StandardScaler()),
        
        # 4. CatBoost регрессор
        ('regressor', CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False
        ))
    ])
    
    print("✅ Pipeline создан с компонентами:")
    for step_name, step_estimator in pipeline.steps:
        print(f"   • {step_name}: {step_estimator.__class__.__name__}")
    
    return pipeline


def save_pipeline(pipeline, filepath, additional_info=None):
    """
    Сохранение обученного pipeline в файл.
    
    Args:
        pipeline: обученный sklearn Pipeline
        filepath: путь для сохранения
        additional_info: дополнительная информация для сохранения
    """
    print(f"\n💾 Сохранение обученного pipeline...")
    
    # Создаем директорию если не существует
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Подготавливаем данные для сохранения
    pipeline_data = {
        'pipeline': pipeline,
        'save_timestamp': pd.Timestamp.now().isoformat(),
        'sklearn_version': joblib.__version__,
        'additional_info': additional_info or {}
    }
    
    # Сохраняем с помощью joblib (оптимально для sklearn объектов)
    joblib.dump(pipeline_data, filepath, compress=3)
    
    # Проверяем размер файла
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
    
    print(f"✅ Pipeline сохранен:")
    print(f"   • Файл: {filepath}")
    print(f"   • Размер: {file_size:.2f} MB")
    print(f"   • Время: {pipeline_data['save_timestamp']}")
    
    return filepath


def load_pipeline(filepath):
    """
    Загрузка обученного pipeline из файла.
    
    Args:
        filepath: путь к файлу с pipeline
        
    Returns:
        tuple: (pipeline, metadata)
    """
    print(f"\n📂 Загрузка обученного pipeline...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pipeline файл не найден: {filepath}")
    
    # Загружаем данные
    pipeline_data = joblib.load(filepath)
    
    pipeline = pipeline_data['pipeline']
    metadata = {
        'save_timestamp': pipeline_data.get('save_timestamp', 'Unknown'),
        'sklearn_version': pipeline_data.get('sklearn_version', 'Unknown'),
        'additional_info': pipeline_data.get('additional_info', {})
    }
    
    print(f"✅ Pipeline загружен:")
    print(f"   • Файл: {filepath}")
    print(f"   • Сохранен: {metadata['save_timestamp']}")
    print(f"   • Компоненты: {len(pipeline.steps)} шагов")
    
    for step_name, step_estimator in pipeline.steps:
        print(f"     - {step_name}: {step_estimator.__class__.__name__}")
    
    return pipeline, metadata


def demonstrate_different_outlier_methods(features_train, target_train, features_test, target_test):
    """
    Демонстрация различных методов обработки выбросов.
    """
    print(f"\n🔬 Сравнение методов обработки выбросов:")
    print("=" * 50)
    
    methods = ['iqr', 'zscore', 'percentile']
    results = {}
    result_saver = ResultSaver()
    
    for method in methods:
        print(f"\n🔍 Тестируем метод: {method.upper()}")
        print("-" * 30)
        
        # Создаем pipeline с конкретным методом
        pipeline = create_ml_pipeline(outlier_method=method, random_state=42)
        
        # Обучение
        print(f"   🎯 Обучение с {method}...")
        pipeline.fit(features_train, target_train)
        
        # Предсказания
        train_pred = pipeline.predict(features_train)
        test_pred = pipeline.predict(features_test)
        
        # Метрики
        train_metrics = result_saver.calculate_metrics(target_train, train_pred)
        test_metrics = result_saver.calculate_metrics(target_test, test_pred)
        
        results[method] = {
            'train_r2': train_metrics['r2'],
            'test_r2': test_metrics['r2'],
            'train_mape': train_metrics['mape'],
            'test_mape': test_metrics['mape'],
            'overfitting': train_metrics['r2'] - test_metrics['r2']
        }
        
        print(f"   📊 Test R²: {test_metrics['r2']:.3f} | Test MAPE: {test_metrics['mape']:.3f}")
    
    # Сравнительная таблица
    print(f"\n📋 Сравнительная таблица методов:")
    print("-" * 60)
    print(f"{'Метод':<12} {'Test R²':<10} {'Test MAPE':<12} {'Переобучение':<12}")
    print("-" * 60)
    
    for method, metrics in results.items():
        print(f"{method.upper():<12} {metrics['test_r2']:<10.3f} {metrics['test_mape']:<12.3f} {metrics['overfitting']:<12.3f}")
    
    # Определяем лучший метод
    best_method = max(results.items(), key=lambda x: x[1]['test_r2'])
    print(f"\n🏆 Лучший метод: {best_method[0].upper()} (Test R² = {best_method[1]['test_r2']:.3f})")
    
    return best_method[0], results


def train_phase(features_train, target_train, best_method, models_dir="models"):
    """
    Фаза обучения модели с сохранением pipeline.
    
    Returns:
        tuple: (pipeline_filepath, train_predictions, data_info)
    """
    print(f"\n🎯 ФАЗА ОБУЧЕНИЯ на {features_train.shape[0]:,} объектах...")
    print("=" * 50)
    
    # Создание финального pipeline с лучшим методом
    pipeline = create_ml_pipeline(outlier_method=best_method, random_state=42)
    
    # Обучение pipeline
    print(f"\n🚀 Обучение pipeline...")
    pipeline.fit(features_train, target_train)
    
    # Предсказания на обучающих данных для анализа
    train_predictions = pipeline.predict(features_train)
    
    # Анализ преобразований по этапам
    print(f"\n🔍 Анализ преобразований:")
    
    # 1. После обработки выбросов
    after_outliers = pipeline.named_steps['outlier_handler'].transform(features_train)
    print(f"   • После outlier handler: {after_outliers.shape}")
    
    # 2. После feature engineering
    after_features = pipeline.named_steps['feature_engineering'].transform(after_outliers)
    print(f"   • После feature engineering: {after_features.shape}")
    
    # 3. После масштабирования
    after_scaling = pipeline.named_steps['scaler'].transform(after_features)
    print(f"   • После scaling: {after_scaling.shape}")
    
    data_info = {
        'original_features': features_train.shape[1],
        'after_outlier_handling': after_outliers.shape[1],
        'after_feature_engineering': after_features.shape[1], 
        'final_features': after_scaling.shape[1],
        'train_samples': features_train.shape[0],
        'feature_expansion_ratio': after_features.shape[1] / features_train.shape[1],
        'best_outlier_method': best_method
    }
    
    # Сохранение обученного pipeline
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    pipeline_filename = f"california_housing_pipeline_{best_method}_{timestamp}.joblib"
    pipeline_filepath = os.path.join(models_dir, pipeline_filename)
    
    additional_info = {
        'dataset': 'California Housing',
        'best_outlier_method': best_method,
        'data_info': data_info
    }
    
    saved_filepath = save_pipeline(pipeline, pipeline_filepath, additional_info)
    
    print(f"\n✅ Обучение завершено:")
    print(f"   • Исходные признаки: {data_info['original_features']}")
    print(f"   • Финальные признаки: {data_info['final_features']}")
    print(f"   • Расширение: {data_info['feature_expansion_ratio']:.1f}x")
    print(f"   • Лучший метод: {best_method.upper()}")
    print(f"   • Pipeline сохранен: {saved_filepath}")
    
    return saved_filepath, train_predictions, data_info


def inference_phase(features_test, pipeline_filepath):
    """
    Фаза инференса с загрузкой сохраненного pipeline.
    
    Args:
        features_test: тестовые данные
        pipeline_filepath: путь к сохраненному pipeline
        
    Returns:
        tuple: (test_predictions, pipeline_metadata)
    """
    print(f"\n🔮 ФАЗА ИНФЕРЕНСА на {features_test.shape[0]:,} объектах...")
    print("=" * 50)
    
    # Загрузка обученного pipeline
    pipeline, metadata = load_pipeline(pipeline_filepath)
    
    # Предсказания на тестовых данных
    print(f"\n🚀 Выполнение предсказаний...")
    test_predictions = pipeline.predict(features_test)
    
    print(f"✅ Инференс завершен:")
    print(f"   • Предсказания получены для всех {len(test_predictions):,} объектов")
    print(f"   • Диапазон предсказаний: {test_predictions.min():.2f} - {test_predictions.max():.2f}")
    print(f"   • Модель обучена: {metadata['save_timestamp']}")
    
    return test_predictions, metadata


def analyze_and_save_results(target_train, train_pred, target_test, test_pred, 
                            data_info, best_method, result_saver):
    """
    Анализ и сохранение результатов эксперимента.
    """
    print(f"\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ...")
    print("=" * 50)
    
    # Вычисление метрик
    train_metrics = result_saver.calculate_metrics(target_train, train_pred)
    test_metrics = result_saver.calculate_metrics(target_test, test_pred)
    
    # Информация о модели
    model_info = {
        'model_type': 'CatBoostRegressor',
        'pipeline_components': [
            f'{best_method.upper()}OutlierHandler',
            'FeatureUnion[PolynomialFeatures, Interactions, Ratios, Log, Binning]',
            'StandardScaler', 
            'CatBoostRegressor'
        ],
        'feature_engineering': {
            'original_features': ['passthrough'],
            'sklearn_components': ['PolynomialFeatures', 'KBinsDiscretizer'],
            'custom_components': ['RatioFeatureGenerator', 'LogFeatureGenerator']
        },
        'outlier_method': best_method,
        'hyperparameters': {
            'iterations': 500,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3
        },
        'pipeline_saved': True
    }
    
    # Дополнительная информация
    additional_info = {
        'dataset': 'California Housing',
        'test_size': 0.2,
        'random_state': 42,
        'specialized_components': True,
        'outlier_methods_compared': ['iqr', 'zscore', 'percentile'],
        'workflow': 'separate_train_inference'
    }
    
    # Сохранение эксперимента
    filepath = result_saver.save_experiment(
        experiment_name='california_separate_workflow',
        model_info=model_info,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        data_info=data_info,
        additional_info=additional_info
    )
    
    # Вывод ключевых метрик
    print(f"\n📈 Ключевые метрики (метод: {best_method.upper()}):")
    print(f"   Train MAPE: {train_metrics['mape']:.3f} | Test MAPE: {test_metrics['mape']:.3f}")
    print(f"   Train R²:   {train_metrics['r2']:.3f} | Test R²:   {test_metrics['r2']:.3f}")
    print(f"   Переобучение R²: {train_metrics['r2'] - test_metrics['r2']:.3f}")
    
    return filepath


def main():
    """
    Главная функция демонстрации раздельных фаз обучения и инференса.
    """
    print("🚀 Демонстрация раздельного workflow: ОБУЧЕНИЕ → СОХРАНЕНИЕ → ИНФЕРЕНС")
    print("Sklearn Pipeline сохраняется после обучения и загружается для инференса")
    print("=" * 80)
    
    # Инициализация
    result_saver = ResultSaver()
    random_state = 42
    
    try:
        # 1. Загрузка данных
        features, target, feature_names, description = load_california_housing()
        
        # 2. Разделение на train/test (имитация разных наборов данных)
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, 
            test_size=0.2, 
            random_state=random_state,
            shuffle=True
        )
        
        print(f"\n📋 Разделение данных:")
        print(f"   • Обучение: {features_train.shape[0]:,} объектов")
        print(f"   • Тест: {features_test.shape[0]:,} объектов (имитация новых данных)")
        
        # 3. ДЕМОНСТРАЦИЯ: Сравнение методов outlier handling
        print("\n" + "=" * 80)
        print("🔬 ЭТАП 1: СРАВНЕНИЕ МЕТОДОВ OUTLIER HANDLING")
        print("=" * 80)
        
        best_method, comparison_results = demonstrate_different_outlier_methods(
            features_train, target_train, features_test, target_test
        )
        
        # 4. ФАЗА ОБУЧЕНИЯ С СОХРАНЕНИЕМ PIPELINE
        print("\n" + "=" * 80)
        print("🎯 ЭТАП 2: ОБУЧЕНИЕ И СОХРАНЕНИЕ PIPELINE")
        print("=" * 80)
        
        pipeline_filepath, train_predictions, data_info = train_phase(
            features_train, target_train, best_method
        )
        
        # 5. ФАЗА ИНФЕРЕНСА С ЗАГРУЗКОЙ PIPELINE
        print("\n" + "=" * 80)
        print("🔮 ЭТАП 3: ЗАГРУЗКА И ИНФЕРЕНС")
        print("=" * 80)
        
        test_predictions, pipeline_metadata = inference_phase(
            features_test, pipeline_filepath
        )
        
        # 6. АНАЛИЗ И СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
        print("\n" + "=" * 80)
        print("📊 ЭТАП 4: АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("=" * 80)
        
        results_filepath = analyze_and_save_results(
            target_train, train_predictions,
            target_test, test_predictions,
            data_info, best_method, result_saver
        )
        
        # 7. ИТОГОВАЯ СВОДКА
        print(f"\n" + "=" * 80)
        print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
        print("=" * 80)
        
        print(f"📁 Сохраненные файлы:")
        print(f"   • Pipeline: {pipeline_filepath}")
        print(f"   • Результаты: {results_filepath}")
        
        print(f"\n🔄 Workflow продемонстрирован:")
        print(f"   1. ✅ Сравнение методов outlier handling")
        print(f"   2. ✅ Обучение и сохранение sklearn Pipeline") 
        print(f"   3. ✅ Загрузка Pipeline и инференс на новых данных")
        print(f"   4. ✅ Полное логирование результатов в JSON")
        
        print(f"\n💡 Практическое применение:")
        print(f"   • Pipeline можно загрузить в продакшене: joblib.load('{os.path.basename(pipeline_filepath)}')")
        print(f"   • Все преобразования сохранены и воспроизводятся автоматически")
        print(f"   • Поддержка кастомных трансформеров из custom_elements.py")
        
    except Exception as e:
        print(f"\n❌ Ошибка выполнения: {str(e)}")
        raise


if __name__ == "__main__":
    main()
