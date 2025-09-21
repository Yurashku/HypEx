# Архитектура HypEx: Подробное руководство

## 1. Введение и философия

### Общая концепция библиотеки

HypEx (Hypothesis Experiments) — это библиотека для проведения статистических экспериментов, построенная на принципах
модульности, расширяемости и многоуровневой абстракции. Основная идея заключается в создании гибкой системы, которая
позволяет как быстро запускать стандартные эксперименты (A/B тесты, A/A тесты, matching), так и конструировать сложные
кастомные пайплайны обработки данных.

Библиотека решает ключевую проблему: разрыв между потребностями бизнес-пользователей, которым нужны готовые решения, и
потребностями исследователей данных, которым требуется гибкость и возможность кастомизации.

### Принцип многоуровневой абстракции

HypEx реализует 8 уровней абстракции, от простого использования готовых решений до модификации ядра библиотеки:

1. **Уровень платформы** — использование через UI без написания кода
2. **Уровень конструктора** — создание сценариев через визуальный интерфейс
3. **Уровень шаблонов** — запуск предконфигурированных сценариев
4. **Уровень оболочек (Shell)** — использование готовых экспериментов в несколько строк кода
5. **Уровень композиции** — создание экспериментов из готовых блоков
6. **Уровень расширения** — создание новых блоков через наследование
7. **Уровень модификации** — глубокие доработки базовых механик
8. **Уровень ядра** — изменение фундаментального поведения

Эта философия пронизывает всю архитектуру: каждый слой системы предоставляет свой уровень абстракции, позволяя
пользователям работать на комфортном для них уровне сложности.

### Основные архитектурные принципы

**1. Композиция над наследованием**

- Эксперименты строятся путем композиции Executor'ов
- Каждый Executor выполняет одну конкретную задачу
- Сложное поведение достигается через комбинацию простых компонентов

**2. Единый поток данных**

- Все данные проходят через ExperimentData
- Каждый Executor может читать и модифицировать данные
- Результаты сохраняются в структурированном виде

**3. Разделение ответственности**

- Executor'ы выполняют вычисления
- Experiment'ы управляют последовательностью выполнения
- Reporter'ы форматируют результаты
- Shell'ы предоставляют удобный интерфейс

**4. Расширяемость через полиморфизм**

- Абстрактные базовые классы определяют контракты
- Конкретные реализации следуют единому интерфейсу
- Новая функциональность добавляется через создание новых классов

**5. Immutability где возможно**

- ExperimentData копируется при необходимости
- Transformers работают с копиями данных
- Состояние изменяется явно и контролируемо

## 2. Обзор архитектуры

### Три основных слоя

Архитектура HypEx построена на трех основных слоях, каждый из которых имеет четкую зону ответственности:

#### Слой Shell (Пользовательский интерфейс)

Самый верхний слой, предоставляющий готовые к использованию решения:

- **ExperimentShell** — базовый класс для всех оболочек
- **AATest, ABTest, HomogeneityTest, Matching** — предконфигурированные эксперименты
- **Output классы** — форматирование и представление результатов

Этот слой скрывает всю сложность и позволяет запускать эксперименты в 2-3 строки кода:

```python
test = ABTest(multitest_method="bonferroni")
results = test.execute(data)
```

#### Слой Experiments (Оркестрация)

Средний слой, отвечающий за управление потоком выполнения:

- **Experiment** — базовый класс для композиции Executor'ов
- **Специализированные эксперименты** — OnRoleExperiment, GroupExperiment, ParamsExperiment
- **Управление итерациями** — CycledExperiment, IfParamsExperiment

Этот слой определяет, КАК и В КАКОМ ПОРЯДКЕ выполняются вычисления.

#### Слой Executors/Reporters (Исполнение и отчетность)

Нижний слой, где происходят фактические вычисления:

- **Executors** — выполняют конкретные операции (тесты, преобразования, анализ)
- **Reporters** — извлекают и форматируют результаты из ExperimentData

Этот слой определяет, ЧТО именно вычисляется и КАК представляются результаты.

### Взаимодействие между слоями

```
Пользователь
    ↓
[Shell Layer]
    - Принимает простые параметры
    - Создает сложную конфигурацию
    - Возвращает отформатированные результаты
    ↓
[Experiments Layer]
    - Управляет последовательностью
    - Координирует выполнение
    - Применяет стратегии (группы, итерации)
    ↓
[Executors/Reporters Layer]
    - Выполняют атомарные операции
    - Модифицируют ExperimentData
    - Форматируют результаты
```

### Поток данных через систему

1. **Инициализация**: Dataset оборачивается в ExperimentData
2. **Выполнение**: Каждый Executor в цепочке:
    - Читает необходимые данные из ExperimentData
    - Выполняет свою операцию
    - Записывает результаты обратно в ExperimentData
3. **Отчетность**: Reporter извлекает результаты и форматирует их
4. **Вывод**: Output классы представляют результаты пользователю

Ключевая особенность — ExperimentData служит общей шиной данных, через которую компоненты обмениваются информацией, не
зная о существовании друг друга.

## 3. Структуры данных: Dataset и ExperimentData

### Dataset: Универсальный контейнер данных

Dataset — это основная структура для хранения табличных данных в HypEx. Он обеспечивает:

#### Архитектура Dataset

**Компоненты:**

- **Backend** — адаптер для работы с конкретной реализацией (PandasBackend)
- **Roles** — словарь, сопоставляющий колонки с их семантическими ролями
- **Методы** — богатый API для манипуляции данными

**Ключевые особенности:**

- Инкапсулирует pandas DataFrame, но может работать с другими backend'ами
- Каждая колонка имеет роль (Role), определяющую её семантику
- Поддерживает цепочки операций (fluent interface)
- Immutable по умолчанию (операции возвращают новый Dataset)

#### Система ролей (ABCRole)

Роли определяют семантическое значение колонок:

```
ABCRole (abstract)
├── InfoRole — информационные поля
├── TargetRole — целевые переменные
├── FeatureRole — признаки
├── TreatmentRole — индикатор группы эксперимента
├── GroupingRole — поле для группировки
├── StratificationRole — поле для стратификации
├── PreTargetRole — baseline значения
├── StatisticRole — статистические метрики
├── FilterRole — фильтрующие поля
├── TempRole — временные роли
│   ├── TempTargetRole
│   ├── TempTreatmentRole
│   └── TempGroupingRole
└── AdditionalRole — дополнительные поля
    ├── AdditionalTargetRole
    ├── AdditionalTreatmentRole
    ├── AdditionalGroupingRole
    └── AdditionalMatchingRole
```

Роли позволяют:

- Автоматически определять, какие колонки использовать для анализа
- Валидировать корректность данных
- Применять правильные преобразования

#### Основные операции Dataset

```python
# Создание
ds = Dataset(data=df, roles={'outcome': TargetRole(), 'group': TreatmentRole()})

# Фильтрация и выборка
ds_filtered = ds[ds['value'] > 100]
ds_subset = ds[['col1', 'col2']]

# Группировка и агрегация
grouped = ds.groupby('category')
aggregated = ds.agg(['mean', 'std'])

# Преобразования
ds_transformed = ds.apply(lambda x: x * 2, role={'result': StatisticRole()})

# Слияние
ds_merged = ds1.merge(ds2, on='id')
```

### ExperimentData: Контекст эксперимента

ExperimentData — это расширенный контейнер, который хранит не только исходные данные, но и все промежуточные результаты,
метаданные и состояние эксперимента.

#### Архитектура ExperimentData

**Пространства имен:**

```python
class ExperimentDataEnum:
    analysis_tables = "analysis_tables"  # Результаты Calculator'ов
    variables = "variables"  # Переменные и настройки
    additional_fields = "additional_fields"  # Дополнительные колонки данных
```

**Компоненты:**

- **ds: Dataset** — основные данные эксперимента
- **analysis_tables: dict** — результаты анализов по ID Executor'ов
- **variables: dict** — переменные и метаданные
- **additional_fields: dict** — дополнительные поля данных

#### Основные операции ExperimentData

```python
# Создание
experiment_data = ExperimentData(dataset)

# Сохранение результата Executor'а
experiment_data.set_value(
    space=ExperimentDataEnum.analysis_tables,
    executor_id="TTest╤reliability 0.05╤",
    value=test_results
)

# Получение результата по ID
result = experiment_data.analysis_tables["TTest╤reliability 0.05╤"]

# Поиск ID'шников по типу Executor'а
ids = experiment_data.get_ids(TTest)

# Добавление дополнительных полей
experiment_data.add_fields({'predicted_score': prediction_column})
```

### Принципы работы с данными

1. **Immutability**: ExperimentData копируется при необходимости изменения основного Dataset
2. **Namespace separation**: Разные типы результатов хранятся в разных пространствах имен
3. **ID-based access**: Результаты индексируются по уникальным ID Executor'ов
4. **Rich metadata**: Поддержка метаданных и переменных для каждого эксперимента

## 4. Ядро системы: Executor Framework

### Концепция Executor

Executor — это фундаментальный строительный блок HypEx. Каждый Executor представляет собой атомарную операцию, которая:

- Принимает ExperimentData на вход
- Выполняет одну конкретную задачу
- Возвращает модифицированный ExperimentData

Это позволяет строить сложные пайплайны из простых, переиспользуемых компонентов.

### Базовый класс Executor

```python
class Executor(ABC):
    def __init__(self, key: Any = ""):
        self._id: str = ""  # Уникальный идентификатор
        self._params_hash = ""  # Хеш параметров
        self.key: Any = key  # Дополнительный ключ
        self._generate_id()

    @abstractmethod
    def execute(self, data: ExperimentData) -> ExperimentData:
        """Основной метод выполнения"""
        raise AbstractMethodError

    def set_params(self, params: dict) -> None:
        """Динамическая установка параметров"""
        # Позволяет изменять параметры после создания

    def _generate_id(self):
        """Генерация уникального ID"""
        # ClassName╤ParamsHash╤Key
```

**Ключевые особенности:**

- **Единый интерфейс** — все Executor'ы реализуют метод `execute`
- **Самоидентификация** — каждый Executor знает свой уникальный ID
- **Параметризация** — поддержка динамического изменения параметров
- **Композируемость** — Executor'ы можно комбинировать в цепочки

### Три ветви наследования

#### 1. Calculator — вычислительная ветвь

Calculator добавляет возможность выполнять вычисления вне контекста ExperimentData:

```python
class Calculator(Executor, ABC):
    @classmethod
    def calc(cls, data: Dataset, **kwargs):
        """Статический метод для вычислений"""
        return cls._inner_function(data, **kwargs)

    @staticmethod
    @abstractmethod
    def _inner_function(data: Dataset, **kwargs) -> Any:
        """Реализация вычисления"""
        pass
```

**Назначение:** Разделение логики вычисления от логики работы с ExperimentData.

#### 2. IfExecutor — условная ветвь

```python
class IfExecutor(Executor):
    def __init__(self, if_executor: Executor, else_executor: Executor = None):
        self.if_executor = if_executor
        self.else_executor = else_executor

    def check_rule(self, data: ExperimentData) -> bool:
        """Логика принятия решения"""
        pass

    def execute(self, data: ExperimentData) -> ExperimentData:
        if self.check_rule(data):
            return self.if_executor.execute(data)
        elif self.else_executor:
            return self.else_executor.execute(data)
        return data
```

**Назначение:** Условное выполнение на основе состояния данных.

#### 3. MLExecutor — машинное обучение ветвь

```python
class MLExecutor(Executor, ABC):
    def fit(self, X: Dataset, Y: Dataset) -> 'MLExecutor':
        """Обучение модели"""
        pass

    def predict(self, X: Dataset) -> Dataset:
        """Предсказания"""
        pass

    def score(self, X: Dataset, Y: Dataset) -> float:
        """Метрика качества"""
        pass
```

**Назначение:** Стандартный интерфейс для ML операций.

### Система ID

Каждый Executor имеет уникальный ID, состоящий из:

```
ClassName╤ParamsHash╤Key
```

**Примеры:**

```python
TTest() → "TTest╤╤"
TTest(reliability=0.01) → "TTest╤rel 0.01╤"
TTest(key="revenue") → "TTest╤╤revenue"
```

### Жизненный цикл Executor

1. **Создание:**
   ```python
   executor = TTest(reliability=0.05)
   # Генерируется ID: "TTest╤╤"
   ```

2. **Конфигурация (опционально):**
   ```python
   executor.set_params({"reliability": 0.01})
   # ID обновляется: "TTest╤rel 0.01╤"
   ```

3. **Выполнение:**
   ```python
   result = executor.execute(experiment_data)
   # Результат сохраняется в ExperimentData под ID executor'а
   ```

4. **Поиск результатов:**
   ```python
   # Найти все результаты TTest
   ids = experiment_data.get_ids(TTest)
   # Получить конкретный результат
   result = experiment_data.analysis_tables[ids[0]]
   ```

### Принципы проектирования Executor'ов

1. **Единая ответственность** — один Executor = одна задача
2. **Независимость** — не должны знать о других Executor'ах
3. **Идемпотентность** — повторное выполнение дает тот же результат
4. **Самодостаточность** — содержат всю логику для своей задачи
5. **Testability** — легко тестировать изолированно

### Расширение через наследование

Создание нового Executor:

```python
class MyCustomTest(StatHypothesisTesting):
    @classmethod
    def _inner_function(cls, data: Dataset, test_data: Dataset) -> Dataset:
        # Реализация кастомного теста
        statistic = calculate_my_statistic(data, test_data)
        p_value = calculate_p_value(statistic)

        return Dataset.from_dict({
            "statistic": statistic,
            "p-value": p_value,
            "pass": p_value < cls.reliability
        })

    def execute(self, data: ExperimentData) -> ExperimentData:
        # Использует базовую логику StatHypothesisTesting
        return super().execute(data)
```

Таким образом, Executor Framework обеспечивает:

- **Модульность** — каждая операция инкапсулирована
- **Переиспользуемость** — Executor'ы можно комбинировать по-разному
- **Расширяемость** — легко добавлять новую функциональность
- **Тестируемость** — каждый компонент можно тестировать отдельно

## 5. Extension Framework

### Концепция Extension'ов

Extension'ы в HypEx представляют собой систему для инкапсуляции backend-специфичных вычислений, которая обеспечивает работу Calculator'ов с различными типами данных (pandas, Spark, и другими) через единый интерфейс.

#### Ключевое архитектурное разделение

**Backend-агностичность vs Backend-специфичность** — это основной принцип разделения ответственности в HypEx:

**Calculator'ы (Backend-агностичные):**
- Работают исключительно через Dataset API
- Не зависят от конкретной реализации backend'а
- Делегируют backend-специфичные вычисления Extension'ам
- Фокусируются на бизнес-логике анализа

**Extension'ы (Backend-специфичные):**
- Инкапсулируют детали работы с конкретными технологиями
- Предоставляют оптимизированные реализации для разных backend'ов
- Изолируют внешние зависимости (numpy, scipy, sklearn и т.д.)

### Архитектура Extension'ов

```python
class Extension(ABC):
    def calc(self, data: Dataset, **kwargs) -> Dataset:
        """Единая точка входа"""
        backend_type = data.backend.__class__.__name__
        
        if backend_type == "PandasBackend":
            return self._calc_pandas(data, **kwargs)
        elif backend_type == "SparkBackend":
            return self._calc_spark(data, **kwargs)
        else:
            raise NotImplementedError(f"Backend {backend_type} not supported")

    @abstractmethod
    def _calc_pandas(self, data: Dataset, **kwargs) -> Dataset:
        """Реализация для pandas"""
        pass

    def _calc_spark(self, data: Dataset, **kwargs) -> Dataset:
        """Реализация для Spark (опционально)"""
        raise NotImplementedError("Spark backend not implemented")
```

### Примеры Extension'ов

#### StatisticalExtension — статистические вычисления

```python
class TTestExtension(Extension):
    def _calc_pandas(self, data: Dataset, grouping_col: str, target_col: str) -> Dataset:
        """Pandas реализация t-теста"""
        from scipy import stats
        
        groups = data.df.groupby(grouping_col)[target_col]
        group1, group2 = [group.values for name, group in groups]
        
        statistic, p_value = stats.ttest_ind(group1, group2)
        
        return Dataset.from_dict({
            "statistic": statistic,
            "p-value": p_value
        })

    def _calc_spark(self, data: Dataset, grouping_col: str, target_col: str) -> Dataset:
        """Spark реализация через SQL"""
        # Реализация через Spark SQL для больших данных
        spark_df = data.backend.df
        # ... Spark-специфичная логика
```

#### MLExtension — машинное обучение

```python
class CholeskyExtension(Extension):
    def _calc_pandas(self, data: Dataset, features: list) -> Dataset:
        """Разложение Холецкого для pandas"""
        import numpy as np
        
        matrix = data.df[features].values
        cholesky = np.linalg.cholesky(matrix)
        
        return Dataset(
            data=pd.DataFrame(cholesky, columns=features),
            roles={col: FeatureRole() for col in features}
        )

    def _calc_spark(self, data: Dataset, features: list) -> Dataset:
        """Spark ML реализация"""
        from pyspark.ml.linalg import Vectors, Matrices
        # ... Spark ML логика
```

### Интеграция с Calculator'ами

Calculator'ы используют Extension'ы как делегаты:

```python
class TTest(Comparator):
    extension = TTestExtension()  # Класс-атрибут

    @classmethod
    def _inner_function(cls, data: Dataset, test_data: Dataset) -> Dataset:
        # Делегируем вычисления Extension'у
        return cls.extension.calc(
            data=data,
            grouping_col=cls._get_grouping_column(data),
            target_col=cls._get_target_column(test_data)
        )
```

### Жизненный цикл Extension'ов

#### 1. Инициализация и выбор backend'а

```python
# При вызове Extension'а происходит автоматический выбор реализации
extension = CholeskyExtension()

# Extension анализирует тип backend'а Dataset'а
pandas_dataset = Dataset(data=pd.DataFrame(...), roles=...)
result = extension.calc(pandas_dataset)  # Вызовет _calc_pandas()

# Для другого backend'а автоматически выберется другая реализация
# spark_dataset = Dataset(data=spark_df, roles=...)
# result = extension.calc(spark_dataset)  # Вызовет _calc_spark()
```

#### 2. Изоляция зависимостей

```python
class ScipyExtension(Extension):
    def _calc_pandas(self, data: Dataset, **kwargs) -> Dataset:
        try:
            from scipy import stats  # Импорт только при использовании
            # ... логика с scipy
        except ImportError:
            raise ImportError("scipy required for this operation")
```

### Преимущества Extension Framework

#### 1. Архитектурная чистота

- **Четкое разделение ответственности:** Calculator'ы для логики, Extension'ы для реализации
- **Backend-агностичность:** Бизнес-логика не зависит от технических деталей
- **Изоляция зависимостей:** Внешние библиотеки не "протекают" в основной код

#### 2. Производительность и масштабируемость

- **Автоматические оптимизации:** Система автоматически выбирает лучшую реализацию
- **Поддержка разных backend'ов:** pandas, Spark, Dask без изменения бизнес-логики
- **Ленивые вычисления:** Некоторые Extension'ы могут использовать ленивые вычисления

#### 3. Гибкость и расширяемость

- **Простое добавление backend'ов:** Новые backend'ы добавляются через Extension'ы
- **Модульность:** Extension'ы можно переиспользовать и комбинировать
- **Эволюция технологий:** Поддержка новых библиотек через Extension'ы

Extension Framework является ключевым архитектурным решением HypEx, которое обеспечивает баланс между простотой использования и техническими возможностями. Он позволяет Calculator'ам оставаться backend-агностичными, при этом используя мощь специализированных библиотек для каждого типа данных.

## 6. Слой вычислений: Comparators, Transformers, Operators

Слой вычислений содержит конкретные реализации Calculator'ов, каждая из которых специализируется на определенном типе
операций. Все они следуют паттерну разделения вычислительной логики от работы с ExperimentData.

### Comparators: Сравнение и тестирование

Comparators отвечают за сравнение групп и проведение статистических тестов. Они имеют сложную иерархию и богатую
функциональность.

#### Иерархия Comparators

```python
Comparator (abstract)
├── StatHypothesisTesting — статистические тесты
│   ├── TTest — t-тест для сравнения средних
│   ├── KSTest — тест Колмогорова-Смирнова для распределений
│   ├── Chi2Test — хи-квадрат для категориальных переменных
│   └── UTest — тест Манна-Уитни (непараметрический)
├── Difference — вычисление разностей
│   ├── GroupDifference — разности между группами
│   └── RelativeDifference — относительные изменения
└── Correlation — корреляционный анализ
    ├── PearsonCorrelation — корреляция Пирсона
    └── SpearmanCorrelation — ранговая корреляция
```

#### Система ролей в Comparators

Comparators автоматически определяют, какие колонки сравнивать, на основе ролей:

- **TreatmentRole / GroupingRole**: Колонки для разбиения на группы
- **TargetRole**: Метрики для сравнения
- **FeatureRole**: Дополнительные признаки для анализа

#### Примеры Comparators

**TTest** — сравнение средних значений:

```python
class TTest(StatHypothesisTesting):
    def __init__(self, reliability: float = 0.05):
        self.reliability = reliability

    @classmethod
    def _inner_function(cls, data: Dataset, test_data: Dataset) -> Dataset:
        # Использует TTestExtension для backend-специфичных вычислений
        return cls.extension.calc(data, test_data, reliability=cls.reliability)
```

**KSTest** — сравнение распределений:

```python
class KSTest(StatHypothesisTesting):
    @classmethod
    def _inner_function(cls, data: Dataset, test_data: Dataset) -> Dataset:
        # Тест на различие распределений между группами
        return cls.extension.calc(data, test_data, test_type="two_sample")
```

### Transformers: Преобразование данных

Transformers изменяют данные, возвращая модифицированную копию Dataset.

#### Архитектура Transformers

```python
class Transformer(Calculator):
    def execute(self, data: ExperimentData) -> ExperimentData:
        # Копируем данные для безопасного изменения
        new_data = ExperimentData(self.calc(data.ds))
        # Переносим метаданные
        new_data.copy_metadata_from(data)
        return new_data

    def calc(self, data: Dataset) -> Dataset:
        target_cols = self._get_target_columns(data)
        return self._inner_function(data, target_cols, **self.params)
```

#### Примеры Transformers

**NaFiller** — заполнение пропусков:

```python
class NaFiller(Transformer):
    def __init__(self, method: str = "mean"):
        self.method = method

    def _inner_function(data: Dataset, target_cols, method) -> Dataset:
        if method in ['ffill', 'bfill']:
            return data.fillna(method=method)
        elif method == 'mean':
            for col in target_cols:
                data[col].fillna(data[col].mean(), inplace=True)
        return data
```

**CategoryAggregator** — агрегация редких категорий:

```python
class CategoryAggregator(Transformer):
    def __init__(self, min_frequency: float = 0.01):
        self.min_frequency = min_frequency

    def _inner_function(data: Dataset, target_cols, min_freq) -> Dataset:
        for col in target_cols:
            value_counts = data[col].value_counts(normalize=True)
            rare_categories = value_counts[value_counts < min_freq].index
            data[col] = data[col].replace(rare_categories, 'Other')
        return data
```

### Encoders: Кодирование признаков

Encoders преобразуют категориальные признаки в числовые. Результат сохраняется в additional_fields.

```python
class Encoder(Calculator):
    def execute(self, data: ExperimentData) -> ExperimentData:
        target_cols = data.ds.search_columns(roles=FeatureRole(),
                                             search_types=[str])
        encoded = self.calc(data=data.ds, target_cols=target_cols)
        # Сохраняем в additional_fields
        return data.set_value(
            space=ExperimentDataEnum.additional_fields,
            executor_id=self._ids_to_names(encoded.columns),
            value=encoded
        )
```

**DummyEncoder** — one-hot encoding:

```python
class DummyEncoder(Encoder):
    def _inner_function(data: Dataset, target_cols) -> Dataset:
        # Использует pandas.get_dummies
        dummies_df = pd.get_dummies(data[target_cols], drop_first=True)
        # Устанавливает роли для новых колонок
        roles = {col: data.roles[col.split('_')[0]] for col in dummies_df.columns}
        return Dataset(data=dummies_df, roles=roles)
```

### GroupOperators: Операции над группами

GroupOperators выполняют специализированные операции над группами данных, часто используемые в matching и causal
inference.

#### Примеры GroupOperators

**SMD (Standardized Mean Difference)** — стандартизированная разность средних:

```python
class SMD(GroupOperator):
    def _inner_function(data: Dataset, control_data: Dataset, treatment_data: Dataset) -> Dataset:
        # Вычисляет Cohen's d для каждого признака
        results = []
        for col in data.columns:
            mean_diff = treatment_data[col].mean() - control_data[col].mean()
            pooled_std = np.sqrt((treatment_data[col].var() + control_data[col].var()) / 2)
            smd = mean_diff / pooled_std
            results.append({"feature": col, "smd": smd})
        
        return Dataset.from_records(results)
```

**Bias** — оценка смещения после matching:

```python
class Bias(GroupOperator):
    def _inner_function(data: Dataset, matched_indices) -> Dataset:
        # Оценивает качество балансировки после matching
        # Возвращает метрики bias по каждому признаку
```

### Принципы проектирования слоя вычислений

1. **Специализация по типам операций** — каждый тип Calculator'а решает свой класс задач
2. **Унификация интерфейсов** — общие паттерны для работы с ролями и данными  
3. **Расширяемость** — легко добавлять новые операции через наследование
4. **Backend-агностичность** — вычисления делегируются Extension'ам
5. **Composability** — Calculator'ы можно комбинировать в complex pipeline'ы

## 7. Analyzer'ы — комплексный анализ результатов

Analyzer'ы представляют собой специальный класс Executor'ов, которые выполняют высокоуровневый анализ результатов
экспериментов. В отличие от простых вычислительных блоков (Calculator'ов), Analyzer'ы работают с результатами множества
других Executor'ов, агрегируют их и принимают комплексные решения.

### Архитектура Analyzer'ов

Analyzer'ы наследуются напрямую от Executor, минуя Calculator, так как их задача — не вычисления над сырыми данными, а
анализ уже полученных результатов. Типичный Analyzer:

1. **Извлекает результаты** предыдущих Executor'ов из ExperimentData
2. **Анализирует и агрегирует** эти результаты согласно своей логике
3. **Принимает решения** на основе комплексных критериев
4. **Сохраняет итоговый анализ** в analysis_tables

### OneAAStatAnalyzer — анализ одного A/A теста

**Назначение:** Анализирует результаты статистических тестов для одного разбиения A/A теста.

**Функциональность:**

- Извлекает результаты всех примененных тестов (TTest, KSTest, Chi2Test)
- Для каждой метрики подсчитывает количество прошедших тестов
- Оценивает общее качество разбиения по pass rate
- Классифицирует качество: excellent (>95%), good (>80%), acceptable (>60%), poor (<60%)

**Входные данные:** Результаты статистических тестов в analysis_tables

**Выходные данные:**

- Статистика по каждой метрике (passed/total tests, pass_rate, status)
- Общая оценка качества разбиения (overall pass_rate, quality level)

### AAScoreAnalyzer — выбор лучшего разбиения

**Назначение:** Анализирует результаты множественных A/A тестов и выбирает оптимальное разбиение.

**Функциональность:**

- Извлекает результаты всех итераций OneAAStatAnalyzer
- Вычисляет score для каждого разбиения согласно критерию:
    - `max_pass_rate` — максимизация доли прошедших тестов
    - `min_bias` — минимизация смещения между группами
    - `balanced` — комбинация pass_rate и баланса групп
- Ранжирует разбиения по качеству
- Восстанавливает параметры лучшего Splitter'а
- Применяет лучшее разбиение к данным

**Входные данные:** Результаты множественных OneAAStatAnalyzer

**Выходные данные:**

- ID и параметры лучшего разбиения
- Статистика по всем разбиениям (scores, mean, std)
- Колонка с лучшим разбиением в additional_fields

### ABAnalyzer — анализ A/B теста

**Назначение:** Выполняет финальный анализ A/B теста с учетом множественного тестирования.

**Функциональность:**

- Проверяет достаточность размера выборки
- Извлекает все p-values из проведенных тестов
- Применяет коррекцию множественного тестирования:
    - Bonferroni — консервативная коррекция
    - Holm — последовательная коррекция
    - FDR — контроль False Discovery Rate
- Оценивает размер эффекта и его confidence interval
- Разделяет статистическую и практическую значимость
- Принимает решение по каждой метрике:
    - `ship` — значимый и практичный эффект
    - `monitor` — значимый, но малый эффект
    - `investigate` — большой, но незначимый эффект
    - `no_effect` — нет эффекта
- Формирует общие рекомендации по эксперименту

**Входные данные:**

- Результаты статистических тестов
- Размеры групп из GroupSizes
- Эффекты из GroupDifference

**Выходные данные:**

- Детальный анализ по каждой метрике
- Скорректированные p-values
- Общее решение и рекомендации
- Оценка рисков и качества выборки

### MatchingAnalyzer — оценка качества matching

**Назначение:** Комплексная оценка качества matching'а между группами.

**Функциональность:**

- Извлекает matched индексы из FaissNearestNeighbors
- Создает matched датасеты для control и treatment
- Оценивает баланс ковариат через SMD (Standardized Mean Difference)
- Вычисляет метрики качества:
    - SMD — стандартизированная разница средних
    - KS статистика — различие распределений
    - Variance ratio — соотношение дисперсий
- Оценивает treatment effect после matching
- Выполняет диагностику проблем:
    - Проверка размера matched выборки
    - Идентификация несбалансированных ковариат
    - Оценка common support region
- Генерирует рекомендации по улучшению

**Входные данные:**

- Matched индексы из additional_fields
- Исходные данные control и treatment групп

**Выходные данные:**

- Количество и доля matched пар
- Баланс по каждой ковариате
- Общие метрики качества matching
- Treatment effect с доверительными интервалами
- Диагностика и рекомендации

### Паттерны использования Analyzer'ов

#### 1. Последовательный анализ

Analyzer'ы размещаются в конце pipeline'а для анализа накопленных результатов:

```python
experiment = Experiment([
    DataPreparation(),
    StatisticalTests(),
    ABAnalyzer(multitest_method="fdr")
])
```

#### 2. Иерархический анализ

Один Analyzer использует результаты другого:

```python
experiment = Experiment([
    ParamsExperiment(...),
    OneAAStatAnalyzer(),  # Анализ каждого теста
    AAScoreAnalyzer()  # Выбор лучшего
])
```

#### 3. Условный анализ

Выбор Analyzer'а на основе характеристик данных:

```python
IfExecutor(
    condition=lambda d: d.n_metrics > 10,
    if_executor=ABAnalyzer(multitest_method="fdr"),
    else_executor=ABAnalyzer(multitest_method=None)
)
```

### Создание кастомных Analyzer'ов

Для создания своего Analyzer'а необходимо:

1. **Наследоваться от Executor** (не от Calculator)
2. **Реализовать метод execute** с логикой:
    - Извлечение нужных результатов через `data.get_ids()`
    - Анализ и агрегация результатов
    - Сохранение через `data.set_value()`
3. **Следовать конвенциям:**
    - Результаты сохранять в analysis_tables
    - Использовать Dataset для структурированных результатов
    - Предоставлять summary и recommendations

### Преимущества Analyzer'ов

1. **Высокоуровневая абстракция** — скрывают сложность анализа за простым интерфейсом
2. **Переиспользуемость** — стандартные паттерны анализа для типовых задач
3. **Композируемость** — можно комбинировать разные виды анализа
4. **Расширяемость** — легко добавлять новые типы анализа
5. **Воспроизводимость** — стандартизированные методы обеспечивают консистентность
6. **Decision-ready** — превращают сырые результаты в actionable insights

Analyzer'ы являются ключевым компонентом для превращения множества технических результатов вычислений в понятные
бизнес-решения и рекомендации.

## 8. Слой экспериментов: Experiment Framework

Слой экспериментов управляет композицией и оркестрацией Executor'ов. Он определяет, КАК и В КАКОМ ПОРЯДКЕ выполняются
операции, предоставляя различные стратегии выполнения.

### Базовый класс Experiment

```python
class Experiment(Executor):
    def __init__(self, executors: Sequence[Executor]):
        self.executors = executors
        super().__init__()

    def execute(self, data: ExperimentData) -> ExperimentData:
        result_data = data
        for executor in self.executors:
            result_data = executor.execute(result_data)
        return result_data

    @property
    def transformer(self) -> bool:
        """Определяет, есть ли среди Executor'ов Transformer'ы"""
        return any(isinstance(ex, Transformer) for ex in self.executors)
```

**Ключевые особенности:**

- **Композиция** — Experiment сам является Executor'ом и может включать другие Experiment'ы
- **Автоопределение копирования** — автоматически определяет, нужно ли копировать данные
- **ID генерация** — создает составной ID из ID своих Executor'ов

### Специализированные эксперименты

#### OnRoleExperiment — применение по ролям

Применяет набор Executor'ов ко всем колонкам с определенной ролью:

```python
class OnRoleExperiment(Experiment):
    def __init__(self, executors: Sequence[Executor], role: ABCRole):
        self.role = role
        super().__init__(executors)

    def execute(self, data: ExperimentData) -> ExperimentData:
        target_columns = data.ds.search_columns(roles=self.role)
        
        result_data = data
        for column in target_columns:
            for executor in self.executors:
                # Применяем executor к каждой колонке с данной ролью
                executor.set_params({"target_column": column})
                result_data = executor.execute(result_data)
        
        return result_data
```

**Применение:**

- Статистические тесты ко всем метрикам
- Применение фильтров ко всем признакам
- Агрегация по всем группирующим колонкам

#### GroupExperiment — обработка по группам

Разбивает данные по группам и применяет Executor'ы к каждой группе:

```python
class GroupExperiment(ExperimentWithReporter):
    def __init__(self,
                 executors: Sequence[Executor],
                 reporter: DatasetReporter,
                 searching_role: ABCRole):
        self.searching_role = searching_role
        super().__init__(executors, reporter)

    def execute(self, data: ExperimentData) -> ExperimentData:
        grouping_columns = data.ds.search_columns(roles=self.searching_role)
        
        all_results = []
        for group_col in grouping_columns:
            for group_value in data.ds[group_col].unique():
                # Фильтруем данные для текущей группы
                group_data = data.ds[data.ds[group_col] == group_value]
                t_data = ExperimentData(group_data)
                
                # Применяем все Executor'ы к группе
                for executor in self.executors:
                    t_data = executor.execute(t_data)
                
                # Сохраняем результат группы
                group_result = self.reporter.report(t_data)
                group_result[group_col] = group_value
                all_results.append(group_result)
        
        return self._set_result(data, all_results, reset_index=False)
```

**Применение:**

- Анализ по сегментам
- Гетерогенные эффекты
- Подгрупповой анализ

#### ParamsExperiment — параметрический поиск

Выполняет эксперимент с различными комбинациями параметров:

```python
class ParamsExperiment(ExperimentWithReporter):
    def __init__(self,
                 executors: Sequence[Executor],
                 reporter: DatasetReporter,
                 params: dict[type, dict[str, Sequence[Any]]]):
        super().__init__(executors, reporter)
        self._params = params
        self._flat_params = []  # Все комбинации параметров

    def _update_flat_params(self):
        """Генерирует все комбинации параметров"""
        # Пример params:
        # {
        #     AASplitter: {
        #         "random_state": [0, 1, 2],
        #         "control_size": [0.4, 0.5, 0.6]
        #     },
        #     TTest: {
        #         "reliability": [0.01, 0.05, 0.1]
        #     }
        # }
        # Создаст 3 * 3 * 3 = 27 комбинаций

        param_combinations = itertools.product(*[
            itertools.product(*[
                itertools.product([param], values)
                for param, values in class_params.items()
            ])
            for class_params in self._params.values()
        ])

        self._flat_params = list(param_combinations)

    def execute(self, data: ExperimentData) -> ExperimentData:
        self._update_flat_params()

        results = []
        for i, flat_param in enumerate(tqdm(self._flat_params)):
            t_data = ExperimentData(data.ds)

            # Применяем параметры к соответствующим Executor'ам
            for executor in self.executors:
                params_for_executor = self._extract_params_for_executor(
                    executor, flat_param
                )
                if params_for_executor:
                    executor.set_params(params_for_executor)

                t_data = executor.execute(t_data)

            # Сохраняем результат итерации
            iteration_result = self.reporter.report(t_data)
            iteration_result["params"] = flat_param
            iteration_result["iteration"] = i
            results.append(iteration_result)

        return self._set_result(data, results)
```

#### IfParamsExperiment — параметрический поиск с условием остановки

Добавляет возможность ранней остановки при достижении условия:

```python
class IfParamsExperiment(ParamsExperiment):
    def __init__(self,
                 executors: Sequence[Executor],
                 reporter: DatasetReporter,
                 params: dict[type, dict[str, Sequence[Any]]],
                 stopping_criterion: IfExecutor):
        super().__init__(executors, reporter, params)
        self.stopping_criterion = stopping_criterion

    def execute(self, data: ExperimentData) -> ExperimentData:
        self._update_flat_params()

        for i, flat_param in enumerate(tqdm(self._flat_params)):
            t_data = ExperimentData(data.ds)

            # Выполняем эксперимент
            for executor in self.executors:
                params_for_executor = self._extract_params_for_executor(
                    executor, flat_param
                )
                if params_for_executor:
                    executor.set_params(params_for_executor)
                t_data = executor.execute(t_data)

            # Проверяем условие остановки
            if_result = self.stopping_criterion.execute(t_data)
            if_executor_id = if_result.get_one_id(
                self.stopping_criterion.__class__,
                ExperimentDataEnum.variables
            )

            if if_result.variables[if_executor_id]["response"]:
                # Условие выполнено - останавливаемся
                final_result = self.reporter.report(t_data)
                final_result["params"] = flat_param
                final_result["iteration"] = i
                return self._set_result(data, [final_result])

        # Условие не выполнено ни для одной комбинации
        return data
```

**Применение:**

- Поиск первого "хорошего" разбиения для A/A теста
- Early stopping в оптимизации
- Адаптивные эксперименты

### Композиция экспериментов

Эксперименты можно вкладывать друг в друга, создавая сложные pipeline'ы:

```python
# Сложный эксперимент для A/A тестирования
aa_experiment = Experiment([
    # 1. Параметрический поиск лучшего разбиения
    ParamsExperiment(
        executors=[
            AASplitter(),
            Experiment([  # Вложенный эксперимент
                GroupSizes(),
                OnRoleExperiment(  # Применить тесты ко всем targets
                    executors=[TTest(), KSTest()],
                    role=TargetRole()
                )
            ])
        ],
        params={
            AASplitter: {"random_state": range(100)}
        },
        reporter=OneAADictReporter()
    ),

    # 2. Анализ результатов
    AAScoreAnalyzer(alpha=0.05),

    # 3. Условное выполнение на основе результатов
    IfAAExecutor(
        if_executor=Experiment([...]),  # Если тест прошел
        else_executor=Experiment([...])  # Если тест не прошел
    )
])
```

### Жизненный цикл Experiment

1. **Создание и конфигурация:**

```python
experiment = Experiment(
    executors=[executor1, executor2, executor3]
)
```

2. **Детекция типа:**

```python
# Автоматически определяет, нужно ли копировать данные
if experiment.transformer:  # True если есть Transformers
# Будет работать с копией данных
```

3. **Выполнение:**

```python
result = experiment.execute(experiment_data)
# Каждый executor выполняется последовательно
# Результаты накапливаются в ExperimentData
```

4. **Доступ к результатам:**

```python
# Experiment может сам сохранить агрегированный результат
experiment_id = experiment.id
aggregated_result = result.analysis_tables[experiment_id]

# Или можно получить результаты отдельных Executor'ов
executor_ids = experiment.get_executor_ids([TTest, KSTest])
```

### Паттерны использования

#### 1. Pipeline паттерн — последовательная обработка

Самый базовый паттерн, где операции выполняются строго последовательно, и каждая использует результаты предыдущей.

```python
# Классический pipeline для A/B теста
ab_pipeline = Experiment([
    # Этап 1: Подготовка данных
    NaFiller(method="ffill"),  # Заполнение пропусков
    OutliersFilter(lower_percentile=0.05),  # Удаление выбросов
    DummyEncoder(),  # Кодирование категорий

    # Этап 2: Основные метрики
    GroupSizes(grouping_role=TreatmentRole()),
    GroupDifference(grouping_role=TreatmentRole()),

    # Этап 3: Статистические тесты
    TTest(grouping_role=TreatmentRole()),
    KSTest(grouping_role=TreatmentRole()),

    # Этап 4: Анализ результатов
    ABAnalyzer(multitest_method="bonferroni")
])

# Можно создавать переиспользуемые подпайплайны
data_preparation = Experiment([
    NaFiller(method="ffill"),
    OutliersFilter(lower_percentile=0.05, upper_percentile=0.95),
    ConstFilter(threshold=0.95),
    CorrFilter(threshold=0.8),
    DummyEncoder()
])

statistical_tests = Experiment([
    TTest(grouping_role=TreatmentRole()),
    KSTest(grouping_role=TreatmentRole()),
    Chi2Test(grouping_role=TreatmentRole())
])

# Композиция подпайплайнов
full_pipeline = Experiment([
    data_preparation,  # Experiment как Executor
    statistical_tests,  # Experiment как Executor
    ABAnalyzer()
])
```

**Преимущества:**

- Простота понимания и отладки
- Легко модифицировать отдельные этапы
- Возможность переиспользования подпайплайнов

#### 2. Branching паттерн — условное ветвление

Выбор пути выполнения на основе характеристик данных или промежуточных результатов.

```python
# Адаптивный выбор теста
adaptive_testing = Experiment([
    # Подготовка
    DataPreparation(),

    # Проверка нормальности
    NormalityTest(),

    # Выбор теста на основе результата
    IfExecutor(
        condition=lambda d: d.normality_test_passed,
        if_executor=TTest(),  # Параметрический тест
        else_executor=UTest()  # Непараметрический тест
    ),

    # Дальнейший анализ
    ResultAnalyzer()
])

# Каскадное ветвление
cascading_analysis = Experiment([
    InitialTest(),

    IfExecutor(
        condition=lambda d: d.p_value < 0.05,
        if_executor=Experiment([
            # Значимый эффект - глубокий анализ
            EffectSizeCalculator(),
            SegmentAnalysis(),
            HeterogeneityTest()
        ]),
        else_executor=Experiment([
            # Незначимый эффект - проверка power
            PowerAnalysis(),
            SampleSizeRecommendation()
        ])
    )
])
```

**Преимущества:**

- Адаптивность к данным
- Оптимизация вычислений
- Автоматический выбор методов

#### 3. Fan-out/Fan-in паттерн — параллельные анализы

Применение разных анализов к одним данным с последующей агрегацией.

```python
# Fan-out: множественные анализы
multi_analysis = Experiment([
    # Подготовка
    DataPreparation(),

    # Fan-out: разные виды анализа
    OnRoleExperiment(
        executors=[
            TTest(),
            KSTest(),
            Chi2Test()
        ],
        role=TargetRole()
    ),

    # Fan-out: анализ по сегментам
    GroupExperiment(
        executors=[
            SegmentStatistics(),
            SegmentTests()
        ],
        searching_role=SegmentRole(),
        reporter=SegmentReporter()
    ),

    # Fan-in: сводная таблица по всем сегментам
    SegmentAggregator()
])
```

**Преимущества:**

- Параллельная обработка независимых анализов
- Ансамблирование результатов для надежности
- Comprehensive анализ с разных углов

#### 4. Grid Search паттерн — поиск оптимальных параметров

Систематический перебор комбинаций параметров для поиска оптимальных.

```python
# Поиск оптимального разбиения для A/A теста
aa_grid_search = ParamsExperiment(
    executors=[
        AASplitter(),  # Будет параметризован
        Experiment([
            GroupSizes(grouping_role=AdditionalTreatmentRole()),
            OnRoleExperiment(  # Применить тесты ко всем targets
                executors=[
                    TTest(grouping_role=AdditionalTreatmentRole()),
                    KSTest(grouping_role=AdditionalTreatmentRole())
                ],
                role=TargetRole()
            )
        ]),
        OneAAStatAnalyzer()
    ],
    params={
        AASplitter: {
            "random_state": range(1000),  # 1000 разных seed'ов
            "control_size": [0.3, 0.5, 0.7],  # 3 варианта размера
            "sample_size": [0.8, 1.0]  # С сэмплированием и без
        }
    },
    reporter=AADictReporter()
)
# Итого: 1000 * 3 * 2 = 6000 комбинаций

# Grid search с ранней остановкой
optimized_search = IfParamsExperiment(
    executors=[...],
    params={
        Splitter: {"random_state": range(10000)},
        Filter: {"threshold": np.linspace(0.01, 0.1, 10)}
    },
    stopping_criterion=IfAAExecutor(
        # Останавливаемся, когда нашли хорошее разбиение
        condition=lambda d: d.aa_score > 0.95
    ),
    reporter=OptimalParamsReporter()
)
```

**Преимущества:**

- Систематический поиск оптимума
- Возможность ранней остановки
- Вложенная оптимизация для сложных pipeline'ов

#### 5. Hierarchical паттерн — иерархическая обработка

Многоуровневая обработка с агрегацией на разных уровнях.

```python
# Иерархический анализ по регионам и городам
hierarchical_analysis = Experiment([
    # Уровень 1: Анализ по регионам
    GroupExperiment(
        executors=[
            # Базовая статистика по региону
            GroupSizes(), GroupDifference(),
            TTest(), KSTest(),
            
            # Уровень 2: Анализ городов внутри региона
            GroupExperiment(
                executors=[
                    GroupSizes(), GroupDifference(),
                    TTest()  # Упрощенный анализ для городов
                ],
                searching_role=CityRole(),
                reporter=CityReporter()
            )
        ],
        searching_role=RegionRole(),
        reporter=RegionReporter()
    ),

    # Агрегация по всем уровням
    HierarchicalAggregator()
])

# Многоуровневая предобработка
hierarchical_preprocessing = Experiment([
    # Глобальная предобработка
    GlobalOutliersFilter(),
    
    # Предобработка по группам
    GroupExperiment(
        executors=[
            LocalOutliersFilter(),  # Локальное удаление выбросов
            GroupSpecificNormalization()  # Нормализация по группе
        ],
        searching_role=TreatmentRole(),
        reporter=PreprocessingReporter()
    ),
    
    # Финальная агрегация
    FinalNormalization()
])
```

**Преимущества:**

- Естественное моделирование иерархических структур
- Агрегация на разных уровнях детализации
- Гибкость в обработке сложных данных

### Композиция и переиспользуемость

Эксперименты спроектированы для максимального переиспользования:

```python
# Стандартные блоки
data_cleaning = Experiment([
    NaFiller(method="ffill"),
    OutliersFilter(lower_percentile=0.05, upper_percentile=0.95)
])

feature_engineering = Experiment([
    CategoryAggregator(min_frequency=0.01),
    DummyEncoder(),
    ConstFilter(threshold=0.95)
])

basic_testing = Experiment([
    GroupSizes(),
    GroupDifference(),
    TTest(),
    KSTest()
])

# Композиция для разных случаев
simple_ab_test = Experiment([
    data_cleaning,
    basic_testing,
    ABAnalyzer()
])

advanced_ab_test = Experiment([
    data_cleaning,
    feature_engineering,
    basic_testing,
    Chi2Test(),  # Дополнительный тест
    UTest(),     # Непараметрический тест
    ABAnalyzer(multitest_method="fdr_bh")
])

matching_analysis = Experiment([
    data_cleaning,
    feature_engineering,
    # Специфичные для matching компоненты
    MahalanobisDistance(),
    FaissNearestNeighbors(),
    MatchingMetrics(),
    MatchingAnalyzer()
])
```

Каждый компонент:

- Независим и может быть заменен
- Использует результаты предыдущих через ExperimentData
- Добавляет свои результаты для последующих

Это обеспечивает максимальную гибкость при построении экспериментов.

## 9. Система Reporter'ов

Reporter'ы отвечают за извлечение, форматирование и представление результатов экспериментов. Они служат мостом между
внутренним представлением данных в ExperimentData и форматом, удобным для пользователя или последующей обработки.

### Архитектура Reporter'ов

Reporter'ы работают по принципу "извлечь и отформатировать":

1. **Извлекают результаты** из различных пространств имен ExperimentData
2. **Агрегируют и структурируют** информацию согласно своей логике
3. **Форматируют вывод** в нужном виде (dict, Dataset, HTML, PDF и т.д.)
4. **Сохраняют семантику** — каждый Reporter знает, какие результаты и как интерпретировать

### Базовый класс Reporter

**Reporter** — абстрактный базовый класс для всех репортеров:

**Контракт:**

- Метод `report(data: ExperimentData)` — единая точка входа
- Возвращает Any — конкретный формат определяется наследником
- Не модифицирует ExperimentData — только читает данные
- Детерминированность — одинаковые данные дают одинаковый отчет

### DictReporter — универсальная основа

DictReporter является фундаментальным классом для большинства Reporter'ов в HypEx. Его философия — предоставить
универсальный промежуточный формат (словарь), который легко преобразовать в любой другой.

#### Концепция DictReporter

**Основная идея:** Все результаты экспериментов можно представить как плоский словарь с уникальными ключами.

**Формат ключей:**

```python
# Технический формат (front=False)
"TTest╤╤revenue" → {"p-value": 0.042, "statistic": 2.15}

# User-friendly формат (front=True)
"revenue_ttest_pvalue" → 0.042
"revenue_ttest_statistic" → 2.15
```

**Преимущества плоской структуры:**

- Отсутствие вложенности упрощает обработку
- Уникальные ключи предотвращают конфликты
- Легко конвертировать в табличный формат
- Простая сериализация (JSON, pickle)

#### Методы извлечения в DictReporter

DictReporter предоставляет набор методов для извлечения разных типов результатов:

- **`extract_from_one_row_dataset()`** — для скалярных результатов
- **`_extract_from_comparators()`** — для результатов сравнений
- **`_get_struct_dict()`** — создание структурированного словаря
- **`_convert_dataset_to_dict()`** — конвертация Dataset в dict

Каждый наследник DictReporter переопределяет метод `report()`, комбинируя эти методы извлечения для создания нужного
словаря.

### OnDictReporter — универсальный форматтер

OnDictReporter — это паттерн Decorator для DictReporter'ов. Он позволяет преобразовать базовый словарный формат в любой
другой без изменения логики извлечения.

#### Архитектура форматирования

```
ExperimentData → DictReporter → dict → OnDictReporter → Любой формат
                     ↑                         ↓
              (извлечение)              (форматирование)
```

**Ключевая идея:** Разделение извлечения данных и их представления.

#### Примеры OnDictReporter

**DatasetReporter** — конвертация в Dataset:

```python
class DatasetReporter(OnDictReporter):
    def __init__(self, dict_reporter: DictReporter):
        self.dict_reporter = dict_reporter

    def report(self, data: ExperimentData) -> Dataset:
        result_dict = self.dict_reporter.report(data)
        return Dataset.from_dict(result_dict)
```

**HTMLReporter** — генерация HTML отчетов:

```python
class HTMLReporter(OnDictReporter):
    def report(self, data: ExperimentData) -> str:
        result_dict = self.dict_reporter.report(data)
        return self._dict_to_html(result_dict)
```

### Специализированные Reporter'ы

#### ABDictReporter — A/B тестирование

```python
class ABDictReporter(TestDictReporter):
    def report(self, data: ExperimentData) -> dict:
        result = {}

        # Извлекаем результаты тестов
        result.update(self._extract_from_comparators(data))

        # Извлекаем размеры групп
        result.update(self._extract_from_executors(data, [GroupSizes]))

        # Извлекаем эффекты
        result.update(self._extract_from_executors(data, [GroupDifference]))

        # Результаты ABAnalyzer (если есть)
        ab_analysis = self._extract_from_executors(data, [ABAnalyzer])
        if ab_analysis:
            result.update(ab_analysis)

        return result
```

#### MatchingDictReporter — Matching анализ

```python
class MatchingDictReporter(DictReporter):
    def report(self, data: ExperimentData) -> dict:
        result = {}

        # Результаты matching
        result.update(self._extract_matching_results(data))

        # Метрики качества
        result.update(self._extract_quality_metrics(data))

        # Bias оценки
        result.update(self._extract_bias_analysis(data))

        return result
```

### Принципы проектирования Reporter'ов

1. **Separation of Concerns:**
    - Извлечение (что достать) отделено от форматирования (как показать)
    - DictReporter отвечает за извлечение
    - OnDictReporter отвечает за форматирование

2. **Composability:**
    - Reporter'ы можно комбинировать
    - Один базовый reporter, множество форматов вывода

3. **Reusability:**
    - Один Reporter может использоваться в разных экспериментах
    - Форматтеры переиспользуются для разных типов данных

4. **Extensibility:**
    - Легко добавить новый формат вывода
    - Не нужно менять логику извлечения

5. **Testability:**
    - Reporter'ы тестируются независимо от экспериментов
    - Форматтеры тестируются отдельно от извлечения

### Преимущества архитектуры Reporter'ов

1. **Гибкость представления** — один эксперимент, множество форматов вывода
2. **Консистентность** — стандартизированные способы извлечения результатов
3. **Масштабируемость** — легко добавлять новые форматы без изменения core
4. **Maintainability** — изменения в форматировании не влияют на логику
5. **User Experience** — пользователь получает результаты в удобном виде

Reporter'ы обеспечивают элегантное решение проблемы представления результатов, позволяя HypEx адаптироваться под
различные use cases и требования к отчетности.

## 10. Shell слой — готовые решения

### Концепция Shell слоя

Shell слой представляет собой вершину архитектуры HypEx, реализуя уровень 4 абстракции — "использование готовых
экспериментов". Этот слой воплощает философию библиотеки о предоставлении простых решений для сложных задач, скрывая всю
внутреннюю сложность за интуитивно понятным интерфейсом.

**Ключевая идея Shell слоя:** Превратить запуск статистического эксперимента из многоэтапного процесса конфигурирования
в простой вызов метода с минимальным набором параметров.

### Двухкомпонентная архитектура Shell слоя

Shell слой построен на принципе разделения ответственности между двумя типами компонентов:

#### ExperimentShell — конструктор экспериментов

ExperimentShell является расширяемым конструктором типовых экспериментов. Его задачи:

- **Инкапсуляция сложности**: Скрывает детали композиции Executor'ов, их последовательность и параметры
- **Интеллектуальная сборка**: Динамически строит конфигурацию эксперимента на основе входных параметров
- **Встроенные лучшие практики**: Автоматически применяет проверенные паттерны для каждого типа эксперимента
- **Валидация и безопасность**: Обеспечивает корректность входных данных и параметров

```python
class ExperimentShell:
    def __init__(self, experiment: Experiment, output: Output):
        self.experiment = experiment  # Композиция Executor'ов
        self.output = output  # Форматтер результатов

    def execute(self, data: Dataset) -> Output:
        # Единый интерфейс для всех экспериментов
        experiment_data = ExperimentData(data)
        result_data = self.experiment.execute(experiment_data)
        self.output.extract(result_data)
        return self.output
```

#### Output — система форматированного представления

Output классы реализуют кураторский подход к представлению результатов:

- **Селективность**: Из всех возможных результатов выбирают только ключевые для принятия решений
- **Контекстуализация**: Представляют результаты с учетом специфики конкретного типа эксперимента
- **Удобство восприятия**: Форматируют данные для максимального удобства бизнес-пользователей
- **Семантическая организация**: Группируют результаты по смыслу через типизированные атрибуты
- **Доступ к детальным данным**: Сохраняют доступ к полным результатам ExperimentData для углубленного анализа

### Готовые эксперименты

#### AATest — автоматизированное A/A тестирование

**Назначение:** Оценка качества разбиения на группы через многократное A/A тестирование с выбором оптимального варианта.

**Архитектура эксперимента:**

```python
# Упрощенная схема AATest
AATest = ExperimentShell(
    experiment=ParamsExperiment([
        AASplitter(control_size=param, random_state=param),
        OnRoleExperiment([
            GroupSizes(), GroupDifference(), TTest(), KSTest(), Chi2Test()
        ]),
        OneAAStatAnalyzer(),
        IfAAExecutor(stop_condition)  # Досрочная остановка при хорошем результате
    ]),
    output=AAOutput()
)
```

**Ключевые возможности:**

- **precision_mode**: Режим повышенной точности с большим числом итераций
- **control_size**: Размер контрольной группы (по умолчанию 50/50)
- **stratification**: Стратифицированное разбиение для улучшения баланса
- **n_iterations**: Количество попыток разбиения
- **sample_size**: Размер выборки для ускорения на больших данных

**Интеллектуальное поведение:**

- Автоматический выбор лучшего разбиения по pass rate статистических тестов
- Досрочная остановка при достижении отличного качества (>95% pass rate)
- Адаптивная стратегия: стратификация включается автоматически при дисбалансе групп

#### ABTest — A/B тестирование с коррекцией

**Назначение:** Статистическое сравнение контрольной и тестовой групп с корректной обработкой множественного
тестирования.

**Архитектура эксперимента:**

```python
# Упрощенная схема ABTest
ABTest = ExperimentShell(
    experiment=Experiment([
        OnRoleExperiment([
            GroupSizes(), GroupDifference(), TTest(), KSTest(), Chi2Test(),
            # + дополнительные тесты по параметру additional_tests
        ]),
        ABAnalyzer(multitest_method=param)
    ]),
    output=ABOutput()
)
```

**Ключевые возможности:**

- **additional_tests**: Дополнительные статистические тесты ("utest", "psi", "chi2")
- **multitest_method**: Метод коррекции множественного тестирования ("bonferroni", "holm", "fdr_bh")
- **reliability**: Уровень значимости (по умолчанию 0.05)

**Интеллектуальное поведение:**

- Автоматическое применение коррекции множественного тестирования
- Адаптивный выбор тестов в зависимости от типов данных
- Интерпретация результатов с учетом поправок на множественность

#### HomogeneityTest — проверка однородности групп

**Назначение:** Валидация корректности разбиения на группы через проверку их статистической однородности.

**Архитектура эксперимента:**

```python
# Схема HomogeneityTest
HomogeneityTest = ExperimentShell(
    experiment=Experiment([
        OnRoleExperiment([
            GroupSizes(), GroupDifference(), TTest(), KSTest(), Chi2Test()
        ]),
        OneAAStatAnalyzer()
    ]),
    output=HomoOutput()
)
```

**Назначение и применение:**

- Проверка качества randomization в экспериментах
- Валидация исторических разбиений
- Диагностика проблем с балансом групп
- Pre-flight проверка перед запуском A/B теста

#### Matching — анализ сопоставления

**Назначение:** Полный цикл matching анализа от поиска пар до оценки качества сопоставления.

**Архитектура эксперимента:**

```python
# Упрощенная схема Matching
Matching = ExperimentShell(
    experiment=Experiment([
        # Подготовка данных
        OutliersFilter(), NaFiller(), DummyEncoder(),
        # Вычисление расстояний
        MahalanobisDistance() if distance == "mahalanobis",
        # Поиск пар
        FaissNearestNeighbors(n_neighbors=1),
        # Оценка качества
        Bias(), MatchingMetrics(metric=param),
        TTest(), KSTest(),  # если quality_tests включены
        MatchingAnalyzer()
    ]),
    output=MatchingOutput()
)
```

**Ключевые возможности:**

- **group_match**: Сопоставление внутри групп или между группами
- **distance**: Метрика расстояния ("mahalanobis", "euclidean", "cosine")
- **metric**: Метрика качества matching ("ate", "att", "atc")
- **bias_estimation**: Оценка bias после сопоставления
- **quality_tests**: Дополнительные статистические тесты качества

### Система Output'ов

Каждый тип эксперимента имеет специализированный Output класс, оптимизированный для представления результатов
конкретного анализа:

#### AAOutput — результаты A/A тестирования

```python
class AAOutput:
    best_split: Dataset  # Лучшее разбиение с метриками качества
    experiments: Dataset  # Статистика по всем итерациям
    aa_score: Dataset  # Итоговый скор качества разбиения
    best_split_statistic: Dataset  # Детальная статистика лучшего разбиения
```

**Кураторский подход:** Пользователь получает готовое к использованию разбиение и полную диагностику его качества.

#### ABOutput — результаты A/B тестирования

```python
class ABOutput:
    resume: Dataset  # Основные результаты тестов
    multitest: Dataset | str  # Результаты с коррекцией множественного тестирования
    sizes: Dataset  # Размеры групп
```

**Кураторский подход:** Акцент на итоговых решениях с учетом коррекции и интерпретации значимости.

#### HomoOutput — результаты проверки однородности

```python
class HomoOutput:
    resume: Dataset  # Сводка по однородности всех метрик
```

**Кураторский подход:** Компактное представление с фокусом на общей оценке качества разбиения.

#### MatchingOutput — результаты matching анализа

```python
class MatchingOutput:
    full_data: Dataset  # Полный датасет с результатами matching
    quality_results: Dataset  # Метрики качества сопоставления
    indexes: Dataset  # Индексы сопоставленных объектов
```

**Кураторский подход:** Разделение на итоговые данные, диагностику качества и техническую информацию.

### Доступ к детальным результатам

Важная особенность системы Output'ов — сохранение доступа к полным результатам эксперимента для случаев, когда требуется
более глубокий анализ:

```python
# Стандартное использование — кураторские результаты
ab_results = ab_test.execute(dataset)
summary = ab_results.resume  # Основные выводы

# Доступ к детальным результатам при необходимости
full_experiment_data = ab_results.experiment_data  # Полный ExperimentData
raw_test_results = ab_results.additional_reporters  # Результаты всех Reporter'ов
detailed_diagnostics = ab_results.get_detailed_analysis()  # Расширенная диагностика
```

**Принципы доступа к сырым данным:**

- **Прозрачность**: Любой результат нижних уровней остается доступным
- **Постепенная детализация**: От summary к details по мере необходимости
- **Техническая диагностика**: Доступ к промежуточным вычислениям для debugging
- **Интеграция с нижними уровнями**: Возможность извлечь данные для композиции с кастомными Executor'ами

### Расширяемость и конфигурирование

#### Принципы расширяемости Shell'ов

Shell'ы спроектированы как расширяемые конструкторы, которые адаптируются под различные варианты типовых экспериментов:

**1. Параметрическая расширяемость**

```python
# Базовое использование
ab_test = ABTest()

# Расширенная конфигурация
ab_test = ABTest(
    additional_tests=["utest", "psi"],  # Дополнительные тесты
    multitest_method="fdr_bh",  # Альтернативная коррекция
    reliability=0.01  # Повышенная строгость
)
```

**2. Адаптивное поведение**

- AATest автоматически включает стратификацию при дисбалансе
- Matching выбирает оптимальный алгоритм в зависимости от размера данных
- ABTest применяет разные стратегии коррекции в зависимости от количества метрик

**3. Композиционная расширяемость**

```python
# Кастомизация через наследование для новых типовых паттернов
class CustomABTest(ABTest):
    def __init__(self, **kwargs):
        super().__init__(
            additional_tests=["bootstrap", "permutation"],
            multitest_method="custom_fdr",
            **kwargs
        )
```

#### Расширяемость Output'ов

Output'ы поддерживают гибкую настройку представления результатов:

**1. Дополнительные Reporter'ы**

```python
# Добавление кастомных форматтеров
output = AAOutput()
output.additional_reporters = {
    'detailed_diagnostics': CustomDiagnosticsReporter(),
    'export_format': ExcelReporter()
}
```

**2. Конфигурация основных Reporter'ов**

- Настройка форматирования чисел (precision, научная нотация)
- Выбор языка для интерпретаций (русский/английский)
- Настройка уровней детализации (краткий/полный отчет)

### Практические примеры использования

#### Минимальный код для типовых сценариев

```python
# A/A тест для валидации разбиения
aa_test = AATest()
aa_results = aa_test.execute(dataset)
print(f"Качество разбиения: {aa_results.aa_score}")

# A/B тест с консервативными настройками
ab_test = ABTest(multitest_method="bonferroni")
ab_results = ab_test.execute(dataset)
print(f"Значимые различия: {ab_results.resume}")

# Проверка однородности групп
homo_test = HomogeneityTest()
homo_results = homo_test.execute(dataset)
print(f"Группы однородны: {homo_results.resume}")

# Matching анализ с расширенной диагностикой
matching = Matching(
    bias_estimation=True,
    quality_tests=["ttest", "kstest"]
)
matching_results = matching.execute(dataset)
print(f"Качество matching: {matching_results.quality_results}")
```

#### Конфигурация под специфические требования

```python
# Высокоточный A/A тест для критических экспериментов
aa_precision = AATest(
    precision_mode=True,  # Увеличенное число итераций
    n_iterations=1000,  # Явное задание количества попыток
    stratification=True,  # Принудительная стратификация
    control_size=0.3  # Нестандартное соотношение групп
)

# A/B тест с множественными гипотезами и строгой коррекцией
ab_comprehensive = ABTest(
    additional_tests=["utest", "bootstrap", "psi"],
    multitest_method="holm",
    reliability=0.01
)

# Matching с кастомной метрикой расстояния
matching_custom = Matching(
    distance="cosine",  # Альтернативная метрика
    metric="att",  # Average Treatment Effect on Treated
    group_match=True  # Matching внутри групп
)
```

### Философия типовых решений

#### Принцип "90% в 2 строчки"

Shell слой реализует ключевой принцип библиотеки HypEx: 90% практических задач должны решаться в 2 строчки кода.

Каждый Shell создавался на основе анализа реальных индустриальных потребностей:

- **AATest** — стандартная процедура валидации разбиений в product экспериментах
- **ABTest** — основной инструмент для измерения эффекта изменений
- **HomogeneityTest** — обязательная проверка для observational studies
- **Matching** — стандартный подход для работы с non-randomized данными

```python
# 90% задач A/B тестирования решается так:
ab_test = ABTest()
results = ab_test.execute(dataset)
```

#### Встроенные лучшие практики

Каждый Shell инкапсулирует проверенные методологические подходы:

**Статистическая корректность:**

- Автоматическая коррекция множественного тестирования в ABTest
- Стратификация для улучшения баланса в AATest
- Проверка assumptions для параметрических тестов
- Robust оценки при наличии выбросов

**Production готовность:**

- Валидация входных данных и параметров
- Graceful обработка edge cases (малые выборки, отсутствующие данные)
- Информативные сообщения об ошибках
- Детерминированность результатов при фиксированном random_state

**Интерпретируемость:**

- Автоматическая генерация текстовых выводов
- Предупреждения о потенциальных проблемах (низкая мощность, нарушение assumptions)
- Рекомендации по улучшению качества анализа

#### Границы применимости и расширения

**Когда использовать Shell'ы:**

- Стандартные экспериментальные сценарии
- Необходимость быстрого получения надежных результатов
- Работа пользователей без глубокой экспертизы в статистике
- Production системы с требованиями к стабильности

**Когда переходить на уровень композиции:**

- Нестандартные экспериментальные дизайны
- Специфические статистические методы, не покрытые библиотекой
- Сложные multi-stage эксперименты
- Исследовательские задачи с неопределенной методологией

**Создание новых Shell'ов:**
Новые Shell'ы оправданы при появлении устойчивых индустриальных паттернов:

- Новый тип эксперимента становится стандартной практикой
- Определенная комбинация методов регулярно воспроизводится
- Есть возможность инкапсулировать лучшие практики для новой области

Shell слой HypEx воплощает идеал современного статистического софтвера: максимальная простота использования при
сохранении методологической строгости и гибкости настройки под конкретные потребности.

## 11. Практические примеры и сценарии

### Философия практических примеров

> **Важное примечание:** Код в данном разделе написан в демонстрационных целях для иллюстрации архитектурных принципов
> HypEx. Примеры не взяты из реальных проектов и не претендуют на переиспользуемость без дополнительной доработки. При
> создании production решений рекомендуется использовать официальную документацию API и лучшие практики разработки.

Данный раздел демонстрирует, как архитектура HypEx работает в реальных сценариях. Каждый пример показывает применение
определенного уровня абстракции и объясняет, почему именно этот подход оптимален для конкретной задачи.

**Принцип эволюции решений:** Мы покажем, как одна и та же бизнес-потребность может решаться на разных уровнях сложности
в зависимости от специфических требований. Это демонстрирует ключевое преимущество архитектуры HypEx — возможность
начать с простого решения и постепенно углубляться по мере необходимости.

**Связь с уровнями абстракции:**

- **Уровень 4 (Shell)** — стандартные сценарии, 90% задач
- **Уровень 5 (Композиция)** — нестандартные комбинации стандартных блоков
- **Уровень 6 (Расширение)** — создание новой функциональности

### Сценарий 1: Стандартный A/B тест (Уровень 4 — Shell)

**Бизнес-задача:** Продуктовая команда тестирует новый дизайн кнопки на конверсию в покупку. Нужно статистически
корректно сравнить контрольную и тестовую группы.

**Характеристики задачи:**

- Типовый A/B тест с одной метрикой
- Стандартные требования к статистической значимости
- Нужен быстрый и надежный результат
- Команда не имеет глубокой экспертизы в статистике

**Решение — использование ABTest Shell:**

```python
from hypex import ABTest
from hypex.dataset import Dataset, TargetRole, TreatmentRole

# Подготовка данных
dataset = Dataset(
    data=experiment_data,
    roles={
        'user_group': TreatmentRole(),  # 'control' или 'test'
        'conversion': TargetRole()  # 0 или 1
    }
)

# Запуск A/B теста
ab_test = ABTest()
results = ab_test.execute(dataset)

# Получение результатов
print("=== РЕЗУЛЬТАТЫ A/B ТЕСТА ===")
print(f"Основные выводы:\n{results.resume}")
print(f"Размеры групп:\n{results.sizes}")
```

**Что происходит под капотом:**

1. ABTest автоматически применяет набор статистических тестов (t-test, KS-test)
2. ABAnalyzer анализирует результаты и формирует выводы
3. ABOutput форматирует результаты в удобном для бизнеса виде

**Результат:**

```
Группа control: 1000 пользователей, конверсия 5.2%
Группа test: 1000 пользователей, конверсия 6.8%
Статистическая значимость: p-value = 0.032 (значимо)
Рекомендация: Внедрять изменение
```

**Почему этот уровень подходит:**

- Задача полностью покрывается стандартной функциональностью
- Встроенные лучшие практики (коррекция, валидация)
- Минимальный код, максимальная надежность
- Готовые к презентации результаты

### Сценарий 2: A/B тест с дополнительными требованиями (Уровень 5 — Композиция)

**Бизнес-задача:** Та же команда хочет провести более углубленный анализ — добавить PSI (Population Stability Index) для
проверки стабильности распределения пользователей между группами и использовать менее консервативную коррекцию
множественного тестирования.

**Характеристики задачи:**

- Нестандартные дополнительные тесты
- Специфические настройки коррекции
- Сохранение стандартной основы A/B теста

**Решение — композиция стандартных компонентов:**

```python
from hypex.experiments import Experiment, OnRoleExperiment
from hypex.executors import PSITest, TTest, KSTest, GroupSizes, GroupDifference
from hypex.analyzers import ABAnalyzer
from hypex.reporters import ABDictReporter

# Построение кастомного A/B теста
enhanced_ab_test = Experiment([
    # Базовые метрики (стандартно)
    GroupSizes(grouping_role=TreatmentRole()),
    GroupDifference(grouping_role=TreatmentRole()),
    
    # Применяем расширенный набор тестов ко всем метрикам
    OnRoleExperiment([
        TTest(grouping_role=TreatmentRole()),
        KSTest(grouping_role=TreatmentRole()),
        PSITest(grouping_role=TreatmentRole())  # Дополнительный тест
    ], role=TargetRole()),
    
    # Анализ с кастомными настройками
    ABAnalyzer(
        multitest_method="fdr_bh",  # Менее консервативная коррекция
        effect_size_threshold=0.02  # Кастомный порог практической значимости
    )
])

# Выполнение
experiment_data = ExperimentData(dataset)
results = enhanced_ab_test.execute(experiment_data)

# Извлечение результатов
reporter = ABDictReporter()
formatted_results = reporter.report(results)
```

**Архитектурные преимущества:**

- **Переиспользование**: Используем стандартные компоненты
- **Гибкость**: Добавляем только нужную функциональность
- **Консистентность**: Сохраняется логика стандартного A/B теста
- **Прозрачность**: Видны все этапы анализа

### Сценарий 3: Создание кастомного Executor (Уровень 6 — Расширение)

**Бизнес-задача:** Data Science команда нуждается в специфическом статистическом тесте — Permutation Test — который
не входит в стандартный набор HypEx.

**Характеристики задачи:**

- Требуется новая функциональность
- Нужна интеграция с существующей архитектурой
- Планируется переиспользование в разных экспериментах

**Решение — создание кастомного Executor:**

```python
from hypex.executors.base import StatHypothesisTesting
from hypex.dataset import Dataset
import numpy as np
from typing import Optional

class PermutationTest(StatHypothesisTesting):
    """
    Permutation Test для сравнения групп без предположений о распределении.
    """
    
    def __init__(self, 
                 n_permutations: int = 10000,
                 reliability: float = 0.05,
                 key: str = ""):
        self.n_permutations = n_permutations
        super().__init__(reliability=reliability, key=key)
    
    @classmethod
    def _inner_function(cls, 
                       data: Dataset, 
                       test_data: Dataset,
                       n_permutations: int = 10000,
                       reliability: float = 0.05) -> Dataset:
        """
        Реализация permutation test.
        """
        # Извлекаем данные для двух групп
        group_col = cls._get_grouping_column(data)
        target_col = cls._get_target_column(test_data)
        
        groups = data.df[group_col].unique()
        if len(groups) != 2:
            raise ValueError("Permutation test поддерживает только 2 группы")
        
        group1_data = test_data.df[data.df[group_col] == groups[0]][target_col]
        group2_data = test_data.df[data.df[group_col] == groups[1]][target_col]
        
        # Наблюдаемая разность средних
        observed_diff = group2_data.mean() - group1_data.mean()
        
        # Объединяем данные для перестановок
        combined_data = np.concatenate([group1_data, group2_data])
        n1, n2 = len(group1_data), len(group2_data)
        
        # Выполняем перестановки
        permutation_diffs = []
        np.random.seed(42)  # Для воспроизводимости
        
        for _ in range(n_permutations):
            # Случайно перемешиваем данные
            np.random.shuffle(combined_data)
            perm_group1 = combined_data[:n1]
            perm_group2 = combined_data[n1:]
            
            # Вычисляем разность для перестановки
            perm_diff = perm_group2.mean() - perm_group1.mean()
            permutation_diffs.append(perm_diff)
        
        # Вычисляем p-value (двусторонний тест)
        permutation_diffs = np.array(permutation_diffs)
        p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))
        
        # Формируем результат
        return Dataset.from_dict({
            "statistic": observed_diff,
            "p-value": p_value,
            "n_permutations": n_permutations,
            "pass": p_value < reliability
        })

# Использование в композиции
advanced_ab_experiment = Experiment([
    GroupSizes(),
    GroupDifference(),
    
    OnRoleExperiment([
        TTest(),           # Стандартный параметрический тест
        KSTest(),          # Стандартный непараметрический тест
        PermutationTest(   # Наш кастомный тест
            n_permutations=20000,
            reliability=0.01
        )
    ], role=TargetRole()),
    
    ABAnalyzer(multitest_method="holm")
])
```

**Архитектурные преимущества:**

- **Интеграция**: Кастомный Executor работает как стандартный
- **Переиспользуемость**: Может использоваться в любых экспериментах
- **Тестируемость**: Легко тестировать изолированно
- **Расширяемость**: Базовый класс предоставляет всю инфраструктуру

### Сценарий 4: Комплексный многоэтапный анализ

**Бизнес-задача:** Исследовательская команда изучает эффект нового алгоритма рекомендаций на удержание пользователей. 
Требуется полный цикл: от A/A валидации данных до matching анализа с sensitivity проверками.

**Характеристики задачи:**

- Многоэтапный процесс анализа
- Комбинация разных типов экспериментов
- Сложная логика принятия решений
- Потребность в детальной диагностике

**Решение — комбинация всех уровней архитектуры:**

```python
# Этап 1: Валидация качества данных (Shell уровень)
aa_validation = AATest(precision_mode=True, n_iterations=500)
aa_results = aa_validation.execute(dataset)

if aa_results.aa_score.quality_level != "excellent":
    print("Предупреждение: качество разбиения недостаточно высокое")
    # Дополнительная диагностика...

# Этап 2: Предварительный A/B тест (Композиция)
preliminary_ab = Experiment([
    GroupSizes(), GroupDifference(),
    OnRoleExperiment([TTest(), KSTest(), PermutationTest()], role=TargetRole()),
    ABAnalyzer(multitest_method="fdr_bh")
])

preliminary_results = preliminary_ab.execute(ExperimentData(dataset))

# Этап 3: Решение о дальнейшем анализе
ab_analyzer_id = preliminary_results.get_ids(ABAnalyzer)[0]
ab_summary = preliminary_results.analysis_tables[ab_analyzer_id]

if ab_summary["overall_decision"] in ["ship", "monitor"]:
    print("Обнаружен эффект. Переходим к matching анализу для causal inference.")
    
    # Этап 4: Matching анализ с кастомными компонентами
    matching_analysis = Experiment([
        # Подготовка данных
        OutliersFilter(lower_percentile=0.05, upper_percentile=0.95),
        NaFiller(method="ffill"),
        DummyEncoder(),

        # Вычисление расстояний
        MahalanobisDistance(grouping_role=TreatmentRole()),

        # Поиск пар
        FaissNearestNeighbors(n_neighbors=1),

        # Оценка качества
        Bias(grouping_role=TreatmentRole()),
        MatchingMetrics(metric="ate"),

        # Статистические тесты
        TTest(compare_by="groups"),
        KSTest(compare_by="groups"),

        # Анализ результатов
        MatchingAnalyzer()
    ])
    
    matching_results = matching_analysis.execute(ExperimentData(dataset))
    
    # Этап 5: Sensitivity анализ (Расширение)
    class SensitivityAnalyzer(Executor):
        """Анализ чувствительности результатов к различным параметрам matching."""
        
        def execute(self, data: ExperimentData) -> ExperimentData:
            # Тестируем разные пороги matching
            sensitivity_results = []
            
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                # Повторяем matching с разными порогами
                threshold_experiment = Experiment([
                    FaissNearestNeighbors(threshold=threshold),
                    MatchingMetrics(metric="ate"),
                    TTest(compare_by="groups")
                ])
                
                threshold_data = ExperimentData(data.ds)
                threshold_result = threshold_experiment.execute(threshold_data)
                
                # Извлекаем ключевые метрики
                metrics_id = threshold_result.get_ids(MatchingMetrics)[0]
                ate_estimate = threshold_result.analysis_tables[metrics_id]["ate"]
                
                sensitivity_results.append({
                    "threshold": threshold,
                    "ate_estimate": ate_estimate,
                    "n_matched": len(threshold_result.additional_fields.get("matched_indices", []))
                })
            
            # Сохраняем результаты sensitivity анализа
            sensitivity_dataset = Dataset.from_records(sensitivity_results)
            return data.set_value(
                space=ExperimentDataEnum.analysis_tables,
                executor_id=self.id,
                value=sensitivity_dataset
            )
    
    # Запуск sensitivity анализа
    sensitivity_analyzer = SensitivityAnalyzer()
    final_results = sensitivity_analyzer.execute(matching_results)
    
    # Этап 6: Comprehensive отчетность
    class ComprehensiveMatchingReporter(DictReporter):
        """Comprehensive reporter для всего анализа."""
        
        def report(self, data: ExperimentData) -> dict:
            base_results = super().report(data)
            
            # Добавляем результаты sensitivity анализа
            sensitivity_id = data.get_ids(SensitivityAnalyzer)[0]
            sensitivity_data = data.analysis_tables[sensitivity_id]
            
            base_results['sensitivity_analysis'] = {
                'stability_assessment': self._assess_stability(sensitivity_data),
                'robust_effect_estimate': self._robust_estimate(sensitivity_data),
                'recommended_threshold': self._recommend_threshold(sensitivity_data)
            }

            # Добавляем качественную оценку
            base_results['quality_assessment'] = self._comprehensive_quality_check(data)

            return base_results
        
        def _assess_stability(self, data: Dataset) -> str:
            # Логика оценки стабильности результатов
            estimates = data.df["ate_estimate"].values
            cv = np.std(estimates) / np.mean(estimates)  # Coefficient of variation
            
            if cv < 0.1:
                return "highly_stable"
            elif cv < 0.2:
                return "stable"
            else:
                return "unstable"
        
        def _robust_estimate(self, data: Dataset) -> float:
            # Робастная оценка (медиана)
            return data.df["ate_estimate"].median()
        
        def _recommend_threshold(self, data: Dataset) -> float:
            # Рекомендуемый порог (баланс качества и покрытия)
            data_df = data.df
            # Простая эвристика: максимизируем произведение покрытия и обратной дисперсии
            coverage = data_df["n_matched"] / data_df["n_matched"].max()
            stability = 1 / (data_df["ate_estimate"].rolling(3).std().fillna(1))
            score = coverage * stability
            
            return data_df.loc[score.idxmax(), "threshold"]
        
        def _comprehensive_quality_check(self, data: ExperimentData) -> str:
            # Комплексная оценка качества всего анализа
            # ... логика проверки всех компонентов
            return "EXCELLENT"  # Упрощено

    comprehensive_reporter = ComprehensiveMatchingReporter()
    final_report = comprehensive_reporter.report(final_results)

else:
    print("Эффект не обнаружен или слишком слаб для matching анализа.")
```

**Результат многоэтапного анализа:**

```
=== COMPREHENSIVE MATCHING ANALYSIS ===

Preprocessing Quality:
  - Outliers removed: 234 observations (2.1%)
  - Missing values imputed: 12 features
  - Feature engineering: 8 new features created

Matching Results:
  - Matched pairs: 3,847 из 4,120 (93.4% coverage)
  - Average match distance: 0.18 (good quality)
  - Bias reduction: 85% (excellent)

Treatment Effect:
  - Retention increase: +8.4% (95% CI: [5.2%, 11.6%])
  - P-value: < 0.001 (highly significant)
  - Effect size: Cohen's d = 0.32 (medium effect)

Sensitivity Analysis:
  - Effect stable across thresholds 0.1-0.4
  - Robust estimate: +8.1% ± 1.2%
  - Recommended threshold: 0.25 (best coverage/quality trade-off)

Quality Assessment: EXCELLENT
  - All balance checks passed
  - Covariate overlap sufficient
  - Results robust to specification choices
```

**Архитектурные преимущества комбинированного подхода:**

1. **Эффективность разработки:** Начали с Shell для быстрой оценки
2. **Модульность:** Каждый этап можно тестировать и отлаживать независимо
3. **Переиспользование:** Кастомные компоненты можно использовать в других проектах
4. **Прозрачность:** Полная видимость всех этапов анализа
5. **Научная строгость:** Sensitivity анализ и множественные проверки качества

### Эволюция решения: От простого к сложному

Рассмотрим, как одна и та же задача решается на разных уровнях сложности:

**Задача:** Измерить влияние нового алгоритма рекомендаций на выручку.

#### Этап 1: Первая итерация (Shell)

```python
# Быстрый A/B тест для первичной оценки
ab_test = ABTest()
initial_results = ab_test.execute(dataset)
# Результат: "Есть положительный эффект +12%, p=0.02"
```

#### Этап 2: Углубленный анализ (Композиция)

```python
# Stakeholder'ы просят добавить анализ подгрупп
segmented_experiment = Experiment([
    # Общий анализ
    OnRoleExperiment([TTest(), KSTest()], role=TargetRole()),

    # Анализ по сегментам пользователей
    GroupExperiment([
        OnRoleExperiment([TTest()], role=TargetRole())
    ], grouping_role=UserSegmentRole()),

    ABAnalyzer(multitest_method="fdr_bh")  # Коррекция для multiple segments
])
```

#### Этап 3: Продвинутая статистика (Расширение)

```python
# Экономисты просят добавить causal inference методы
class DoublyRobustEstimator(Calculator):
    """Doubly robust estimation для более точной causal inference"""
    # ... реализация


advanced_experiment = Experiment([
    # Стандартные тесты
    OnRoleExperiment([TTest(), KSTest()], role=TargetRole()),

    # Продвинутые causal methods
    DoublyRobustEstimator(),
    InstrumentalVariablesAnalysis(),

    # Sensitivity анализ
    ConfoundingSensitivityTest(),
])
```

**Ключевой инсайт:** Архитектура HypEx позволяет начать с простого решения и органично наращивать сложность без
переписывания кода. Каждый уровень строится на предыдущем, добавляя необходимую функциональность.

### Руководство по выбору подходящего уровня

#### Принципы принятия решений

**Используйте Shell (Уровень 4), когда:**

- Задача полностью покрывается стандартным экспериментом
- Команда не имеет глубокой экспертизы в статистике
- Нужен быстрый и надежный результат
- Требуется production-ready решение

**Переходите на Композицию (Уровень 5), когда:**

- Нужны дополнительные тесты или метрики
- Требуется нестандартная последовательность операций
- Необходимо объединить несколько типовых анализов
- Хотите больше контроля над процессом

**Создавайте Расширения (Уровень 6), когда:**

- Требуется функциональность, отсутствующая в библиотеке
- У команды есть экспертиза для создания кастомных методов
- Планируется многократное переиспользование
- Нужна интеграция с внешними библиотеками

#### Критерии качественного решения

HypEx позволяет органично развивать решение по мере роста требований.

**Правило 90-9-1:**

- 90% задач решаются Shell'ами (уровень 4)
- 9% требуют композиции стандартных блоков (уровень 5)
- 1% нуждается в создании новой функциональности (уровень 6)

**Критерии качественного решения:**

1. **Простота:** Используйте минимально необходимый уровень сложности
2. **Переиспользование:** Предпочитайте стандартные компоненты кастомным
3. **Модульность:** Разбивайте сложную логику на композируемые части
4. **Тестируемость:** Каждый компонент должен быть независимо тестируемым
5. **Документированность:** Кастомные компоненты требуют подробного описания

Архитектура HypEx спроектирована так, чтобы естественным образом направлять разработчиков к качественным решениям,
предоставляя правильные абстракции для каждого уровня сложности.

## Заключение

### Ключевые архитектурные достижения

Архитектура HypEx успешно решает фундаментальную проблему разрыва между простотой использования и гибкостью статистических инструментов. Ключевые достижения:

**1. Многоуровневая абстракция как архитектурный принцип**

Система 8 уровней абстракции позволяет пользователям работать на комфортном уровне сложности, при этом сохраняя возможность перехода на более детальные уровни по мере роста потребностей. Это обеспечивает:
- Низкий порог входа для начинающих
- Неограниченную гибкость для экспертов
- Естественную эволюцию решений

**2. Композиция как основа расширяемости**

Принцип "композиция над наследованием" реализован последовательно на всех уровнях:
- Executor'ы как атомарные операции
- Experiment'ы как оркестраторы
- Shell'ы как готовые решения
- Каждый компонент может быть заменен или дополнен

**3. Единый поток данных через ExperimentData**

ExperimentData служит универсальной шиной данных, обеспечивая:
- Прозрачность всех промежуточных результатов
- Возможность отладки на любом этапе
- Простую интеграцию новых компонентов
- Воспроизводимость экспериментов

### Принципы, выдержавшие проверку практикой

**Разделение ответственности**

Четкое разделение между вычислениями (Calculator'ы), оркестрацией (Experiment'ы), анализом (Analyzer'ы) и представлением (Reporter'ы) обеспечивает:
- Простоту тестирования
- Легкость модификации
- Возможность независимого развития компонентов

**Backend-агностичность через Extension Framework**

Разделение бизнес-логики и технических деталей реализации позволяет:
- Использовать оптимальные инструменты для каждого типа данных
- Добавлять новые backend'ы без изменения core логики
- Изолировать внешние зависимости

**Кураторский подход к результатам**

Система Output'ов и Reporter'ов превращает технические результаты в business-ready решения:
- 90% пользователей получают готовые выводы
- 10% экспертов имеют доступ к детальным данным
- Консистентная интерпретация результатов

### Архитектурные паттерны HypEx

**Паттерн "Эволюционного усложнения"**

Возможность начать с простого Shell'а и постепенно добавлять сложность:
```
Shell → Композиция → Расширение → Модификация
```

**Паттерн "Декомпозиции сложности"**

Разбиение сложных операций на цепочки простых:
```
Сложный анализ = Preprocessing + Tests + Analysis + Reporting
```

**Паттерн "Полиморфной замещаемости"**

Любой компонент может быть заменен на альтернативную реализацию:
```
TTest ← → PermutationTest ← → BootstrapTest
```

### Практические выводы для разработчиков

**Выбор правильного уровня абстракции**

- Начинайте с Shell'ов для стандартных задач
- Переходите к композиции при нестандартных требованиях
- Создавайте расширения только при необходимости многократного использования

**Принципы качественного кода в HypEx**

1. **Следуйте правилу 90-9-1**: большинство задач должно решаться стандартными инструментами
2. **Предпочитайте композицию наследованию**: комбинируйте готовые блоки
3. **Тестируйте на уровне компонентов**: каждый Executor должен быть протестирован изолированно
4. **Документируйте кастомные решения**: нестандартные компоненты требуют подробного описания

**Архитектурные anti-patterns**

- Создание monolithic Executor'ов, выполняющих множество задач
- Прямые зависимости между Executor'ами
- Обход системы ролей при работе с данными
- Игнорирование Extension Framework при работе с внешними библиотеками

### Направления развития архитектуры

**Масштабируемость**

- Поддержка распределенных вычислений через новые backend'ы
- Оптимизация для больших данных
- Асинхронное выполнение Executor'ов

**Интеграция**

- Расширение Extension Framework для новых ML библиотек
- Интеграция с workflow системами (Airflow, Prefect)
- API для интеграции с внешними системами

**Пользовательский опыт**

- Интеллектуальные рекомендации по выбору методов
- Автоматическая диагностика проблем в данных
- Интерактивные визуализации результатов

### Философия архитектурных решений

HypEx демонстрирует, что хорошая архитектура должна быть:

**Приспособляемой** — легко адаптироваться к изменяющимся требованиям
**Интуитивной** — естественно направлять пользователей к правильным решениям  
**Прозрачной** — позволять понимать и контролировать происходящие процессы
**Расширяемой** — предоставлять четкие точки для добавления новой функциональности

Архитектура HypEx служит примером того, как сложная предметная область (статистический анализ) может быть организована в интуитивно понятную и мощную систему через правильное применение принципов объектно-ориентированного проектирования и многоуровневой абстракции.

Успех HypEx в решении задач статистического анализа демонстрирует универсальность этих архитектурных принципов и их применимость к другим сложным предметным областям.
