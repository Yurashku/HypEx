# Архитектура HypEx: Подробное руководство

## 1. Введение и философия

### Общая концепция библиотеки

HypEx (Hypothesis Experiments) — это библиотека для проведения статистических экспериментов, построенная на принципах модульности, расширяемости и многоуровневой абстракции. Основная идея заключается в создании гибкой системы, которая позволяет как быстро запускать стандартные эксперименты (A/B тесты, A/A тесты, matching), так и конструировать сложные кастомные пайплайны обработки данных.

Библиотека решает ключевую проблему: разрыв между потребностями бизнес-пользователей, которым нужны готовые решения, и потребностями исследователей данных, которым требуется гибкость и возможность кастомизации.

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

Эта философия пронизывает всю архитектуру: каждый слой системы предоставляет свой уровень абстракции, позволяя пользователям работать на комфортном для них уровне сложности.

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

Ключевая особенность — ExperimentData служит общей шиной данных, через которую компоненты обмениваются информацией, не зная о существовании друг друга.

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

ExperimentData — это расширенный контейнер, который хранит не только исходные данные, но и все промежуточные результаты, метаданные и состояние эксперимента.

#### Структура ExperimentData

```python
class ExperimentData:
    # Основные данные
    _data: Dataset  # Исходный датасет
    
    # Дополнительные поля (новые колонки, созданные Executor'ами)
    additional_fields: Dataset
    
    # Переменные (скалярные значения, метрики)
    variables: dict[str, dict[str, Any]]
    
    # Группы (разбиение данных на подгруппы)
    groups: dict[str, dict[str, Dataset]]
    
    # Таблицы анализа (результаты тестов и анализов)
    analysis_tables: dict[str, Dataset]
```

#### Пространства имен (ExperimentDataEnum)

ExperimentData организует данные в четыре пространства имен:

1. **additional_fields** — новые признаки и колонки
   - Результаты encoding'а
   - Вычисленные features
   - Matched индексы

2. **variables** — скалярные значения и словари
   - Параметры моделей
   - Вычисленные константы
   - Метрики качества

3. **groups** — сгруппированные данные
   - Разбиение на control/test
   - Подгруппы для анализа
   - Результаты стратификации

4. **analysis_tables** — результаты анализов
   - Результаты статистических тестов
   - Таблицы сравнений
   - Агрегированные метрики

#### Взаимодействие с Executors

Каждый Executor работает с ExperimentData по следующему паттерну:

```python
def execute(self, data: ExperimentData) -> ExperimentData:
    # 1. Извлечение необходимых данных
    input_data = data.ds  # или data.additional_fields, data.groups и т.д.
    
    # 2. Выполнение операции
    result = self._inner_function(input_data)
    
    # 3. Сохранение результата в нужное пространство имен
    return data.set_value(
        space=ExperimentDataEnum.analysis_tables,
        executor_id=self.id,
        value=result
    )
```

#### Система идентификаторов

Каждый Executor имеет уникальный ID, построенный по схеме:
```
ClassName╤ParamsHash╤Key
```

Где:
- `ClassName` — имя класса Executor'а
- `ParamsHash` — хеш параметров
- `Key` — дополнительный ключ (например, имя колонки)

Это позволяет:
- Избегать повторных вычислений
- Находить результаты конкретных Executor'ов
- Строить зависимости между компонентами
- Восстанавливать состояние Executor

### Поток преобразования данных

```
Dataset → ExperimentData → [Executor1] → ExperimentData' → [Executor2] → ExperimentData''
                              ↓                                ↓
                        Модификация                      Модификация
                     additional_fields               analysis_tables
```

Каждый Executor может:
- Читать из любого пространства имен
- Писать в одно или несколько пространств
- Использовать результаты предыдущих Executor'ов
- Не влиять на исходные данные (immutability)

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
        self._id: str = ""          # Уникальный идентификатор
        self._params_hash = ""      # Хеш параметров
        self.key: Any = key         # Дополнительный ключ
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

**Назначение:** Разделение логики вычисления от логики работы с ExperimentData. Это позволяет:
- Использовать вычисления отдельно от эксперимента
- Тестировать логику изолированно
- Переиспользовать код в разных контекстах

**Основные подклассы Calculator:**

```
Calculator
├── Comparator — сравнение групп и статистические тесты
│   ├── GroupDifference, GroupSizes, PSI
│   ├── StatHypothesisTesting (TTest, KSTest, UTest, Chi2Test)
│   └── PowerTesting (MDEBySize)
├── Transformer — преобразование данных
│   ├── Filters (ConstFilter, CorrFilter, CVFilter, NanFilter, OutliersFilter)
│   ├── NaFiller — заполнение пропусков
│   ├── CategoryAggregator — агрегация категорий
│   └── Shuffle — перемешивание данных
├── Encoder — кодирование категориальных переменных
│   └── DummyEncoder — one-hot encoding
├── GroupOperator — операции над группами
│   ├── SMD — Standardized Mean Difference
│   ├── Bias — оценка смещения
│   └── MatchingMetrics — метрики matching'а
├── MLExecutor — машинное обучение
│   └── FaissNearestNeighbors — поиск ближайших соседей
└── Splitter — разделение на группы
    ├── AASplitter — базовое разделение для A/A теста
    └── AASplitterWithStratification — со стратификацией
```

#### 2. IfExecutor — условное выполнение

IfExecutor реализует паттерн условного выполнения:

```python
class IfExecutor(Executor, ABC):
    def __init__(self, 
                 if_executor: Executor | None = None,
                 else_executor: Executor | None = None):
        self.if_executor = if_executor
        self.else_executor = else_executor
    
    @abstractmethod
    def check_rule(self, data) -> bool:
        """Проверка условия"""
        pass
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        if self.check_rule(data):
            return self.if_executor.execute(data) if self.if_executor else data
        else:
            return self.else_executor.execute(data) if self.else_executor else data
```

**Назначение:** Ветвление логики выполнения на основе условий.

**Пример использования — IfAAExecutor:**
```python
class IfAAExecutor(IfExecutor):
    def check_rule(self, data) -> bool:
        # Проверяет, прошел ли A/A тест
        score_table = data.analysis_tables[...]
        feature_pass = sum([...])  # Подсчет прошедших тестов
        return feature_pass >= 1
```

#### 3. Прямые наследники — Analyzers

Analyzers наследуются напрямую от Executor и выполняют сложный анализ результатов:

```python
# Прямые наследники Executor
├── OneAAStatAnalyzer — анализ статистики одного A/A теста
├── AAScoreAnalyzer — оценка качества A/A тестов и выбор лучшего
├── ABAnalyzer — анализ A/B теста с коррекцией множественного тестирования
└── MatchingAnalyzer — анализ качества matching'а
```

**Особенности Analyzers:**
- Работают с результатами других Executor'ов
- Агрегируют и анализируют множественные результаты
- Принимают сложные решения на основе статистики

### Паттерн "Цепочка ответственности"

Executor'ы образуют цепочку, где каждый:
1. Получает ExperimentData от предыдущего
2. Выполняет свою операцию
3. Передает обогащенный ExperimentData следующему

```python
# Пример цепочки для A/B теста
chain = [
    GroupSizes(),           # Подсчет размеров групп
    GroupDifference(),      # Вычисление разницы между группами
    TTest(),               # T-тест
    KSTest(),              # KS-тест
    ABAnalyzer()           # Анализ и коррекция множественного тестирования
]

# Выполнение цепочки
data = ExperimentData(dataset)
for executor in chain:
    data = executor.execute(data)
```

### Система идентификации и хеширования

Каждый Executor имеет уникальный ID вида: `ClassName╤ParamsHash╤Key`

**Генерация ParamsHash:**
```python
def _generate_params_hash(self):
    # Для AASplitter
    hash_parts = []
    if self.control_size != 0.5:
        hash_parts.append(f"cs {self.control_size}")
    if self.random_state is not None:
        hash_parts.append(f"rs {self.random_state}")
    self._params_hash = "|".join(hash_parts)
```

**Восстановление из ID:**
```python
@classmethod
def build_from_id(cls, executor_id: str):
    """Восстанавливает Executor из его ID"""
    splitted_id = executor_id.split(ID_SPLIT_SYMBOL)
    result = cls()
    result.init_from_hash(splitted_id[1])  # ParamsHash
    return result
```

Это позволяет:
- **Кеширование** — не выполнять повторно одинаковые операции
- **Трассировка** — понимать, какой Executor создал какой результат
- **Воспроизводимость** — восстанавливать состояние из ID

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

## 5. Слой вычислений: Comparators, Transformers, Operators

Слой вычислений содержит конкретные реализации Calculator'ов, каждая из которых специализируется на определенном типе операций. Все они следуют паттерну разделения вычислительной логики от работы с ExperimentData.

### Comparators: Сравнение и тестирование

Comparators отвечают за сравнение групп и проведение статистических тестов. Они имеют сложную иерархию и богатую функциональность.

#### Архитектура Comparator

```python
class Comparator(Calculator, ABC):
    def __init__(self,
                 compare_by: Literal["groups", "columns", "columns_in_groups", "cross"],
                 grouping_role: ABCRole = GroupingRole(),
                 target_roles: ABCRole | list[ABCRole] = TargetRole(),
                 baseline_role: ABCRole = PreTargetRole()):
        self.compare_by = compare_by
        self.grouping_role = grouping_role
        self.target_roles = target_roles
        self.baseline_role = baseline_role
```

**Режимы сравнения (compare_by):**

#### Режим "groups" — сравнение между группами
Самый распространенный режим для A/B тестов. Сравнивает одну и ту же метрику между разными группами (control vs test).

```python
# Данные:
# | user_id | group   | revenue | retention |
# |---------|---------|---------|-----------|
# | 1       | control | 100     | 1         |
# | 2       | test    | 150     | 1         |
# | 3       | control | 80      | 0         |
# | 4       | test    | 120     | 1         |

comparator = TTest(
    compare_by="groups",
    grouping_role=TreatmentRole(),  # группировка по "group"
    target_roles=TargetRole()        # анализ "revenue" и "retention"
)

# Результат:
# Для revenue: сравнение control_revenue vs test_revenue
# Для retention: сравнение control_retention vs test_retention
```

#### Режим "columns" — сравнение колонок между собой
Сравнивает разные метрики внутри всего датасета. Полезно для анализа корреляций или изменений между pre/post периодами.

```python
# Данные:
# | user_id | pre_revenue | post_revenue | pre_clicks | post_clicks |
# |---------|-------------|--------------|------------|-------------|
# | 1       | 100         | 150          | 10         | 15          |
# | 2       | 80          | 90           | 8          | 12          |

comparator = GroupDifference(
    compare_by="columns",
    baseline_role=PreTargetRole(),   # baseline: "pre_revenue", "pre_clicks"
    target_roles=TargetRole()         # сравнить с: "post_revenue", "post_clicks"
)

# Результат:
# Сравнение pre_revenue vs post_revenue (для всех пользователей)
# Сравнение pre_clicks vs post_clicks (для всех пользователей)
```

#### Режим "columns_in_groups" — сравнение колонок внутри каждой группы
Комбинация первых двух режимов. Сравнивает разные метрики, но отдельно для каждой группы.

```python
# Данные:
# | user_id | group   | before_treatment | after_treatment |
# |---------|---------|------------------|-----------------|
# | 1       | control | 100              | 105             |
# | 2       | test    | 100              | 150             |
# | 3       | control | 80               | 82              |
# | 4       | test    | 80               | 120             |

comparator = TTest(
    compare_by="columns_in_groups",
    grouping_role=TreatmentRole(),      # группировка по "group"
    baseline_role=PreTargetRole(),      # baseline: "before_treatment"
    target_roles=TargetRole()           # сравнить с: "after_treatment"
)

# Результат:
# Для группы control: сравнение before_treatment vs after_treatment
# Для группы test: сравнение before_treatment vs after_treatment
# Позволяет оценить эффект внутри каждой группы отдельно
```

#### Режим "cross" — перекрестное сравнение
Самый сложный режим. Берет baseline из одной группы (обычно control) и сравнивает с метриками из других групп. Используется для difference-in-differences анализа.

```python
# Данные:
# | user_id | group   | metric_A | metric_B |
# |---------|---------|----------|----------|
# | 1       | control | 100      | 50       |
# | 2       | test1   | 120      | 60       |
# | 3       | control | 90       | 45       |
# | 4       | test2   | 130      | 70       |

comparator = TTest(
    compare_by="cross",
    grouping_role=TreatmentRole(),      # группировка по "group"
    baseline_role=PreTargetRole(),      # baseline из control: "metric_A"
    target_roles=TargetRole()           # сравнить с метриками из test групп
)

# Результат:
# control.metric_A vs test1.metric_A
# control.metric_A vs test1.metric_B
# control.metric_A vs test2.metric_A
# control.metric_A vs test2.metric_B

# Это позволяет оценить, насколько изменения в test группах
# отличаются от baseline в control группе
```

**Практическое применение режима "cross":**

Режим "cross" особенно полезен для:
1. **Difference-in-Differences (DiD)** — оценка причинно-следственной связи
2. **Synthetic control** — когда control группа служит базой для сравнения
3. **Multiple treatment arms** — когда есть несколько вариантов воздействия

```python
# Пример DiD анализа
# До внедрения фичи:
# control: revenue = 100, retention = 0.5
# test: revenue = 100, retention = 0.5

# После внедрения фичи (только в test):
# control: revenue = 110, retention = 0.52 (естественный рост)
# test: revenue = 140, retention = 0.65 (рост + эффект фичи)

# С режимом "cross" можно оценить:
# (test_after - test_before) - (control_after - control_before)
# = (140 - 100) - (110 - 100) = 30 - истинный эффект фичи
```

#### Базовые метрики (GroupDifference, GroupSizes)

**GroupDifference** вычисляет разницу между группами:
```python
class GroupDifference(Comparator):
    def _inner_function(cls, data: Dataset, test_data: Dataset) -> dict:
        control_mean = data.mean()
        test_mean = test_data.mean()
        return {
            "control mean": control_mean,
            "test mean": test_mean,
            "difference": test_mean - control_mean,
            "difference %": (test_mean / control_mean - 1) * 100
        }
```

**GroupSizes** подсчитывает размеры групп:
```python
class GroupSizes(Comparator):
    def _inner_function(cls, data: Dataset, test_data: Dataset) -> dict:
        size_a = len(data)
        size_b = len(test_data)
        return {
            "control size": size_a,
            "test size": size_b,
            "control size %": (size_a / (size_a + size_b)) * 100,
            "test size %": (size_b / (size_a + size_b)) * 100
        }
```

#### Статистические тесты (StatHypothesisTesting)

Абстрактный класс StatHypothesisTesting добавляет концепцию статистической значимости:

```python
class StatHypothesisTesting(Comparator, ABC):
    def __init__(self, reliability: float = 0.05, **kwargs):
        self.reliability = reliability  # Уровень значимости
        super().__init__(**kwargs)
```

**Конкретные реализации:**

1. **TTest** — t-тест Стьюдента для нормальных распределений
2. **KSTest** — тест Колмогорова-Смирнова для любых распределений
3. **UTest** — U-тест Манна-Уитни для ненормальных распределений
4. **Chi2Test** — хи-квадрат тест для категориальных переменных

Каждый тест возвращает стандартизированный результат:
```python
{
    "p-value": 0.042,
    "statistic": 2.15,
    "pass": True  # p-value < reliability
}
```

#### Специализированные метрики

**PSI (Population Stability Index)** — измеряет стабильность распределения:
```python
class PSI(Comparator):
    def _inner_function(cls, data: Dataset, test_data: Dataset) -> dict:
        # Разбиение на бакеты и вычисление PSI
        psi = sum((y - x) * np.log(y / x) for x, y in zip(data_psi, test_data_psi))
        return {"PSI": psi}
```

**MahalanobisDistance** — вычисляет расстояние Махаланобиса:
```python
class MahalanobisDistance(Calculator):
    def _inner_function(cls, data: Dataset, test_data: Dataset) -> dict:
        # Вычисление ковариационной матрицы
        cov = (data.cov() + test_data.cov()) / 2
        # Преобразование Холецкого и вычисление расстояния
        cholesky = CholeskyExtension().calc(cov)
        mahalanobis_transform = InverseExtension().calc(cholesky)
        y_control = data.dot(mahalanobis_transform.transpose())
        y_test = test_data.dot(mahalanobis_transform.transpose())
        return {"control": y_control, "test": y_test}
```

### Transformers: Преобразование данных

Transformers изменяют сам Dataset (в отличие от других Calculator'ов, которые только вычисляют результаты). Они работают с копией данных для обеспечения immutability.

#### Базовый класс Transformer

```python
class Transformer(Calculator, ABC):
    @property
    def _is_transformer(self) -> bool:
        return True  # Маркер для Experiment
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        # Создает копию и модифицирует данные
        return data.copy(data=self.calc(data=data.ds))
```

#### Фильтры данных

Фильтры изменяют роли колонок или удаляют строки/колонки:

**ConstFilter** — фильтрует константные колонки:
```python
class ConstFilter(Transformer):
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
    
    def _inner_function(data: Dataset, target_cols, threshold) -> Dataset:
        for column in target_cols:
            value_counts = data[column].value_counts(normalize=True)
            if value_counts.iloc[0] > threshold:
                data.roles[column] = InfoRole()  # Понижение роли
        return data
```

**CorrFilter** — удаляет коррелированные признаки:
```python
class CorrFilter(Transformer):
    def _inner_function(data: Dataset, threshold: float = 0.8) -> Dataset:
        corr_matrix = data.corr()
        for col1, col2 in high_corr_pairs:
            # Удаляем признак с меньшей вариативностью
            if data[col1].cv() < data[col2].cv():
                data.roles[col1] = InfoRole()
        return data
```

**OutliersFilter** — фильтрует выбросы по процентилям:
```python
class OutliersFilter(Transformer):
    def __init__(self, lower_percentile: float = 0.05, upper_percentile: float = 0.95):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
    
    def _inner_function(data: Dataset, target_cols, lower, upper) -> Dataset:
        mask = (data[target_cols] < data[target_cols].quantile(lower)) | \
               (data[target_cols] > data[target_cols].quantile(upper))
        return data.drop(data[mask].index)
```

#### Обработка данных

**NaFiller** — заполнение пропущенных значений:
```python
class NaFiller(Transformer):
    def __init__(self, method: Literal["ffill", "bfill"] = None, values=None):
        self.method = method
        self.values = values
    
    def _inner_function(data: Dataset, target_cols, method, values) -> Dataset:
        for column in target_cols:
            data[column] = data[column].fillna(values=values, method=method)
        return data
```

**CategoryAggregator** — объединение редких категорий:
```python
class CategoryAggregator(Transformer):
    def __init__(self, threshold: int = 15, new_group_name: str = "Other"):
        self.threshold = threshold
        self.new_group_name = new_group_name
    
    def _inner_function(data: Dataset, target_cols, threshold, new_name) -> Dataset:
        for column in target_cols:
            value_counts = data[column].value_counts()
            rare_values = value_counts[value_counts < threshold].index
            data[column] = data[column].replace(rare_values, new_name)
        return data
```

**Shuffle** — перемешивание данных:
```python
class Shuffle(Transformer):
    def __init__(self, random_state: int = None):
        self.random_state = random_state
    
    def _inner_function(data: Dataset, random_state) -> Dataset:
        return data.sample(frac=1, random_state=random_state)
```

### Encoders: Кодирование категориальных переменных

Encoders преобразуют категориальные переменные в числовые. Результат сохраняется в additional_fields.

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

GroupOperators выполняют специализированные операции над группами данных, часто используемые в matching и causal inference.

#### SMD (Standardized Mean Difference)

Стандартизированная разница средних — метрика баланса ковариат:
```python
class SMD(GroupOperator):
    def _inner_function(cls, data: Dataset, test_data: Dataset) -> float:
        return (data.mean() - test_data.mean()) / data.std()
```

#### Bias — оценка смещения

Оценивает смещение при matching:
```python
class Bias(GroupOperator):
    @staticmethod
    def calc_coefficients(X: Dataset, Y: Dataset) -> list[float]:
        # Линейная регрессия для оценки коэффициентов
        return np.linalg.lstsq(X.values, Y.values, rcond=-1)[0]
    
    @staticmethod
    def calc_bias(X: Dataset, X_matched: Dataset, coefficients: list[float]) -> list[float]:
        # Вычисление смещения на основе разницы признаков
        return [(j - i).dot(coefficients) for i, j in zip(X.values, X_matched.values)]
```

#### MatchingMetrics — метрики для matching

Вычисляет различные метрики treatment effect:
```python
class MatchingMetrics(GroupOperator):
    def __init__(self, metric: Literal["atc", "att", "ate"] = "ate"):
        self.metric = metric  # Average Treatment on Controls/Treated/Everyone
    
    def _inner_function(cls, data, test_data, target_fields, metric, bias) -> dict:
        # Вычисление Individual Treatment Effect
        itt = test_data[target_fields[0]] - test_data[target_fields[1]]
        itc = data[target_fields[1]] - data[target_fields[0]]
        
        # Коррекция на смещение
        if bias:
            itt -= bias["test"]
            itc -= bias["control"]
        
        # Вычисление метрик с учетом весов
        att = itt.mean()
        atc = itc.mean()
        ate = (att * len(test_data) + atc * len(data)) / (len(test_data) + len(data))
        
        # Вычисление стандартных ошибок и p-values
        return {
            "ATT": [att, se_att, p_val_att, ci_lower_att, ci_upper_att],
            "ATC": [atc, se_atc, p_val_atc, ci_lower_atc, ci_upper_atc],
            "ATE": [ate, se_ate, p_val_ate, ci_lower_ate, ci_upper_ate]
        }
```

### MLExecutors: Машинное обучение

MLExecutors интегрируют алгоритмы машинного обучения в pipeline экспериментов.

#### Базовый класс MLExecutor

```python
class MLExecutor(Calculator, ABC):
    @abstractmethod
    def fit(self, X: Dataset, Y: Dataset = None) -> MLExecutor:
        """Обучение модели"""
        pass
    
    @abstractmethod
    def predict(self, X: Dataset) -> Dataset:
        """Предсказание"""
        pass
    
    def score(self, X: Dataset, Y: Dataset) -> float:
        """Оценка качества"""
        pass
```

#### FaissNearestNeighbors — поиск ближайших соседей

Использует библиотеку FAISS для эффективного поиска ближайших соседей:
```python
class FaissNearestNeighbors(MLExecutor):
    def __init__(self, 
                 n_neighbors: int = 1,
                 two_sides: bool = False,  # Искать пары в обе стороны
                 test_pairs: bool = False,  # Пары для test группы
                 faiss_mode: Literal["base", "fast", "auto"] = "auto"):
        self.n_neighbors = n_neighbors
        self.faiss_mode = faiss_mode
    
    def fit(self, X: Dataset) -> MLExecutor:
        # Создание индекса FAISS
        self.index = faiss.IndexFlatL2(X.shape[1])
        if len(X) > 1_000_000 and self.faiss_mode in ["auto", "fast"]:
            # Используем приближенный поиск для больших данных
            self.index = faiss.IndexIVFFlat(self.index, 1, 1000)
            self.index.train(X.values)
        self.index.add(X.values)
        return self
    
    def predict(self, X: Dataset) -> Dataset:
        # Поиск ближайших соседей
        distances, indices = self.index.search(X.values, self.n_neighbors)
        return Dataset.from_dict({"indices": indices})
```

### Splitters: Разделение на группы

Splitters отвечают за разделение данных на группы для A/A и A/B тестов.

#### AASplitter — базовое разделение

```python
class AASplitter(Calculator):
    def __init__(self,
                 control_size: float = 0.5,
                 random_state: int = None,
                 sample_size: float = None):  # Доля от общих данных
        self.control_size = control_size
        self.random_state = random_state
        self.sample_size = sample_size
    
    def _inner_function(data: Dataset, control_size, random_state, sample_size) -> list[str]:
        # Сэмплирование если нужно
        if sample_size:
            data = data.sample(frac=sample_size, random_state=random_state)
        
        # Разделение на control/test
        n_control = int(len(data) * control_size)
        indices = data.sample(frac=1, random_state=random_state).index
        
        split = pd.Series("test", index=data.index)
        split[indices[:n_control]] = "control"
        
        return split.tolist()
```

#### AASplitterWithStratification — стратифицированное разделение

```python
class AASplitterWithStratification(AASplitter):
    def _inner_function(data: Dataset, control_size, random_state, grouping_fields) -> Dataset:
        if not grouping_fields:
            return super()._inner_function(data, control_size, random_state)
        
        # Разделение внутри каждой страты
        result = []
        for group, group_data in data.groupby(grouping_fields):
            split = super()._inner_function(group_data, control_size, random_state)
            result.extend(split)
        
        return Dataset.from_dict({"split": result}, roles={"split": TreatmentRole()})
```

### Принципы проектирования Calculator'ов

1. **Разделение вычисления и контекста:**
   - `_inner_function` — чистая логика вычисления
   - `execute` — работа с ExperimentData
   - `calc` — статический интерфейс для использования вне экспериментов

2. **Унифицированный результат:**
   - Всегда возвращают Dataset или dict
   - Результаты имеют стандартную структуру
   - Роли определяют семантику результатов

3. **Конфигурируемость:**
   - Параметры передаются через конструктор
   - Поддержка `set_params` для динамического изменения
   - Параметры влияют на ID для кеширования

4. **Поиск подходящих данных:**
   - Используют роли для поиска нужных колонок
   - Могут фильтровать по типам данных
   - Работают с temporary roles при необходимости

### Взаимодействие компонентов

```python
# Пример pipeline для matching
pipeline = [
    # 1. Подготовка данных
    OutliersFilter(lower_percentile=0.05, upper_percentile=0.95),
    NaFiller(method="ffill"),
    DummyEncoder(),
    
    # 2. Вычисление расстояний
    MahalanobisDistance(grouping_role=TreatmentRole()),
    
    # 3. Поиск пар
    FaissNearestNeighbors(n_neighbors=1),
    
    # 4. Оценка качества
    Bias(grouping_role=TreatmentRole()),
    MatchingMetrics(metric="ate"),
    
    # 5. Статистические тесты
    TTest(compare_by="groups"),
    KSTest(compare_by="groups")
]
```

Каждый компонент:
- Независим и может быть заменен
- Использует результаты предыдущих через ExperimentData
- Добавляет свои результаты для последующих

Это обеспечивает максимальную гибкость при построении экспериментов.

## 6. Слой экспериментов: Experiment Framework

Слой экспериментов управляет композицией и оркестрацией Executor'ов. Он определяет, КАК и В КАКОМ ПОРЯДКЕ выполняются операции, предоставляя различные стратегии выполнения.

### Базовый класс Experiment

Experiment — это контейнер для цепочки Executor'ов с логикой управления их выполнением:

```python
class Experiment(Executor):
    def __init__(self,
                 executors: Sequence[Executor],
                 transformer: bool = None,
                 key: Any = ""):
        self.executors = executors
        self.transformer = transformer or self._detect_transformer()
        super().__init__(key)
    
    def _detect_transformer(self) -> bool:
        """Автоматически определяет, содержит ли цепочка Transformer'ы"""
        return all(executor._is_transformer for executor in self.executors)
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        """Последовательное выполнение всех Executor'ов"""
        experiment_data = deepcopy(data) if self.transformer else data
        for executor in self.executors:
            executor.key = self.key
            experiment_data = executor.execute(experiment_data)
        return experiment_data
```

**Ключевые особенности:**
- **Композиция** — объединяет множество Executor'ов
- **Порядок важен** — Executor'ы выполняются последовательно
- **Управление состоянием** — копирует данные если есть Transformer'ы
- **Наследование от Executor** — Experiment сам является Executor'ом (паттерн Composite)

### OnRoleExperiment — выполнение для каждой роли

OnRoleExperiment применяет цепочку Executor'ов к каждой колонке с определенной ролью:

```python
class OnRoleExperiment(Experiment):
    def __init__(self,
                 executors: list[Executor],
                 role: ABCRole | Sequence[ABCRole]):
        self.role = [role] if isinstance(role, ABCRole) else list(role)
        super().__init__(executors)
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        # Находим все колонки с нужной ролью
        for field in data.ds.search_columns(self.role):
            # Устанавливаем временную роль для текущей колонки
            data.ds.tmp_roles = {field: TempTargetRole()}
            # Выполняем всю цепочку для этой колонки
            data = super().execute(data)
            # Очищаем временную роль
            data.ds.tmp_roles = {}
        return data
```

**Пример использования:**
```python
# Применить тесты ко всем target метрикам
experiment = OnRoleExperiment(
    executors=[
        GroupDifference(grouping_role=TreatmentRole()),
        TTest(grouping_role=TreatmentRole()),
        KSTest(grouping_role=TreatmentRole())
    ],
    role=TargetRole()  # Будет применено к каждой колонке с TargetRole
)

# Если есть колонки: revenue (TargetRole), retention (TargetRole), clicks (FeatureRole)
# То тесты будут применены к revenue и retention, но не к clicks
```

### ExperimentWithReporter — эксперименты с отчетностью

Добавляет возможность генерации отчетов после выполнения:

```python
class ExperimentWithReporter(Experiment):
    def __init__(self,
                 executors: Sequence[Executor],
                 reporter: Reporter):
        super().__init__(executors)
        self.reporter = reporter
    
    def one_iteration(self, 
                     data: ExperimentData, 
                     key: str = "") -> Dataset:
        """Одна итерация эксперимента с отчетом"""
        t_data = ExperimentData(data.ds)
        self.key = key
        t_data = super().execute(t_data)
        return self.reporter.report(t_data)
```

Это базовый класс для специализированных экспериментов, которые нуждаются в форматированном выводе результатов.

### CycledExperiment — многократное выполнение

Выполняет эксперимент заданное количество раз (например, для оценки стабильности):

```python
class CycledExperiment(ExperimentWithReporter):
    def __init__(self,
                 executors: list[Executor],
                 reporter: DatasetReporter,
                 n_iterations: int):
        super().__init__(executors, reporter)
        self.n_iterations = n_iterations
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        results = []
        for i in tqdm(range(self.n_iterations)):
            # Каждая итерация начинается с чистых данных
            result = self.one_iteration(data, str(i))
            results.append(result)
        
        # Объединяем результаты всех итераций
        final_result = results[0].append(results[1:])
        return self._set_value(data, final_result)
```

**Применение:**
- Оценка вариативности метрик
- Bootstrap анализ
- Проверка устойчивости результатов

### GroupExperiment — выполнение по группам

Применяет эксперимент отдельно к каждой группе данных:

```python
class GroupExperiment(ExperimentWithReporter):
    def __init__(self,
                 executors: Sequence[Executor],
                 reporter: Reporter,
                 searching_role: ABCRole = GroupingRole()):
        self.searching_role = searching_role
        super().__init__(executors, reporter)
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field = data.ds.search_columns(self.searching_role)
        results = []
        
        # Применяем эксперимент к каждой группе отдельно
        for group, group_data in data.ds.groupby(group_field):
            result = self.one_iteration(
                ExperimentData(group_data), 
                str(group[0])
            )
            results.append(result)
        
        # Объединяем результаты с сохранением индексов групп
        return self._set_result(data, results, reset_index=False)
```

**Применение:**
- Анализ по сегментам
- Гетерогенные эффекты
- Подгрупповой анализ

### ParamsExperiment — параметрический поиск

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
        
        self._flat_params = [
            {class_: dict(params) for class_, params in combination}
            for combination in param_combinations
        ]
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        self._update_flat_params()
        results = []
        
        for flat_param in tqdm(self._flat_params):
            t_data = ExperimentData(data.ds)
            
            # Устанавливаем параметры для каждого Executor
            for executor in self.executors:
                executor.set_params(flat_param)
                t_data = executor.execute(t_data)
            
            # Собираем отчет для этой комбинации
            report = self.reporter.report(t_data)
            results.append(report)
        
        return self._set_result(data, results)
```

**Применение:**
- Grid search для оптимальных параметров
- Sensitivity analysis
- A/A тестирование с разными random_state

**Пример использования:**
```python
# A/A тест с 2000 различными разбиениями
params_exp = ParamsExperiment(
    executors=[
        AASplitter(),  # Будет параметризован
        GroupSizes(grouping_role=AdditionalTreatmentRole()),
        TTest(grouping_role=AdditionalTreatmentRole())
    ],
    params={
        AASplitter: {
            "random_state": range(2000),  # 2000 разных разбиений
            "control_size": [0.5]
        }
    },
    reporter=AADictReporter()
)
```

### IfParamsExperiment — параметрический поиск с ранней остановкой

Расширение ParamsExperiment с возможностью остановки при выполнении условия:

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
        
        for flat_param in tqdm(self._flat_params):
            t_data = ExperimentData(data.ds)
            
            # Выполняем эксперимент
            for executor in self.executors:
                executor.set_params(flat_param)
                t_data = executor.execute(t_data)
            
            # Проверяем условие остановки
            if_result = self.stopping_criterion.execute(t_data)
            if_executor_id = if_result.get_one_id(
                self.stopping_criterion.__class__, 
                ExperimentDataEnum.variables
            )
            
            if if_result.variables[if_executor_id]["response"]:
                # Условие выполнено - останавливаемся
                return self._set_result(data, [self.reporter.report(t_data)])
        
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
    NaFiller(method="ffill"),              # Заполнение пропусков
    OutliersFilter(lower_percentile=0.05), # Удаление выбросов
    DummyEncoder(),                         # Кодирование категорий
    
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
    data_preparation,   # Experiment как Executor
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
# Адаптивный выбор теста на основе характеристик данных
class DataCharacterizer(Executor):
    def execute(self, data: ExperimentData) -> ExperimentData:
        # Анализируем характеристики данных
        is_normal = self._check_normality(data.ds)
        sample_size = len(data.ds)
        has_categories = self._has_categorical(data.ds)
        
        # Сохраняем характеристики
        return data.set_value(
            ExperimentDataEnum.variables,
            self.id,
            {
                "is_normal": is_normal,
                "sample_size": sample_size,
                "has_categories": has_categories
            }
        )

class AdaptiveTestSelector(IfExecutor):
    def check_rule(self, data: ExperimentData) -> bool:
        chars = data.variables[data.get_one_id(DataCharacterizer)]
        # Выбираем подходящий тест
        return chars["is_normal"] and chars["sample_size"] > 30

adaptive_testing = Experiment([
    DataCharacterizer(),
    AdaptiveTestSelector(
        # Для нормальных данных с большой выборкой
        if_executor=Experiment([
            TTest(grouping_role=TreatmentRole()),
            PowerTesting(significance=0.95)
        ]),
        # Для ненормальных или малых выборок
        else_executor=Experiment([
            UTest(grouping_role=TreatmentRole()),
            KSTest(grouping_role=TreatmentRole())
        ])
    )
])

# Многоуровневое ветвление для matching
matching_pipeline = Experiment([
    # Проверка качества данных
    DataQualityChecker(),
    
    IfExecutor(
        condition=lambda d: d.quality_score > 0.9,
        if_executor=Experiment([
            # Высокое качество - используем точный matching
            MahalanobisDistance(),
            FaissNearestNeighbors(n_neighbors=1, faiss_mode="base")
        ]),
        else_executor=IfExecutor(
            condition=lambda d: d.quality_score > 0.5,
            if_executor=Experiment([
                # Среднее качество - используем приближенный matching
                FaissNearestNeighbors(n_neighbors=3, faiss_mode="fast")
            ]),
            else_executor=Experiment([
                # Низкое качество - используем propensity score
                PropensityScoreMatching()
            ])
        )
    )
])
```

**Преимущества:**
- Адаптация к характеристикам данных
- Оптимизация производительности
- Обработка edge cases

#### 3. Fan-out/Fan-in паттерн — параллельное применение

Применение разных анализов к одним данным с последующей агрегацией результатов.

```python
# Fan-out: применяем разные тесты ко всем метрикам
fanout_tests = OnRoleExperiment(
    executors=[
        # Ветка 1: Параметрические тесты
        Experiment([
            TTest(grouping_role=TreatmentRole()),
            GroupDifference(grouping_role=TreatmentRole())
        ]),
        
        # Ветка 2: Непараметрические тесты  
        Experiment([
            KSTest(grouping_role=TreatmentRole()),
            UTest(grouping_role=TreatmentRole())
        ]),
        
        # Ветка 3: Анализ мощности
        Experiment([
            PowerTesting(significance=0.95, power=0.8),
            MDEBySize()
        ])
    ],
    role=TargetRole()  # Применить ко всем target метрикам
)

# Fan-in: агрегация результатов разных тестов
class TestAggregator(Executor):
    def execute(self, data: ExperimentData) -> ExperimentData:
        # Собираем результаты всех тестов
        ttest_results = data.analysis_tables[data.get_one_id(TTest)]
        kstest_results = data.analysis_tables[data.get_one_id(KSTest)]
        utest_results = data.analysis_tables[data.get_one_id(UTest)]
        
        # Агрегируем (например, голосованием)
        aggregated = self._aggregate_by_voting([
            ttest_results, kstest_results, utest_results
        ])
        
        return data.set_value(
            ExperimentDataEnum.analysis_tables,
            self.id,
            aggregated
        )

# Полный fan-out/fan-in pipeline
ensemble_testing = Experiment([
    fanout_tests,      # Fan-out
    TestAggregator()   # Fan-in
])

# Параллельный анализ по сегментам
segment_analysis = Experiment([
    # Fan-out по разным сегментам
    GroupExperiment(
        executors=[
            GroupDifference(),
            TTest()
        ],
        searching_role=SegmentRole(),  # age_group, region, etc.
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
            OnRoleExperiment(
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
            "random_state": range(1000),      # 1000 разных seed'ов
            "control_size": [0.3, 0.5, 0.7],  # 3 варианта размера
            "sample_size": [0.8, 1.0]         # С сэмплированием и без
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

# Вложенный grid search для hyperparameter tuning
hyperparameter_tuning = ParamsExperiment(
    executors=[
        # Внешний уровень: параметры предобработки
        ParamsExperiment(
            executors=[
                OutliersFilter(),
                NaFiller()
            ],
            params={
                OutliersFilter: {
                    "lower_percentile": [0.01, 0.05],
                    "upper_percentile": [0.95, 0.99]
                },
                NaFiller: {
                    "method": ["ffill", "bfill", "mean"]
                }
            },
            reporter=PreprocessingReporter()
        ),
        
        # Внутренний уровень: параметры модели
        ParamsExperiment(
            executors=[
                FaissNearestNeighbors()
            ],
            params={
                FaissNearestNeighbors: {
                    "n_neighbors": [1, 3, 5, 7],
                    "faiss_mode": ["base", "fast"]
                }
            },
            reporter=ModelReporter()
        )
    ],
    params={},  # Внешний уровень без дополнительных параметров
    reporter=HyperparameterReporter()
)
```

**Преимущества:**
- Систематический поиск оптимума
- Возможность ранней остановки
- Вложенная оптимизация для сложных pipeline'ов

#### 5. Hierarchical паттерн — иерархическая обработка

Многоуровневая обработка с агрегацией на разных уровнях.

```python
# Иерархический анализ: пользователь -> сегмент -> общий
hierarchical_analysis = Experiment([
    # Уровень 1: Анализ на уровне пользователей
    OnRoleExperiment(
        executors=[
            UserLevelMetrics(),
            UserLevelTests()
        ],
        role=UserRole()
    ),
    
    # Уровень 2: Агрегация по сегментам
    GroupExperiment(
        executors=[
            SegmentAggregator(),
            SegmentLevelTests()
        ],
        searching_role=SegmentRole(),
        reporter=SegmentReporter()
    ),
    
    # Уровень 3: Общая агрегация
    Experiment([
        GlobalAggregator(),
        GlobalSignificanceTest(),
        MultipleTestingCorrection()
    ])
])

# Каскадный анализ с фильтрацией на каждом уровне
cascade_filtering = Experiment([
    # Этап 1: Грубая фильтрация
    ConstFilter(threshold=0.99),
    
    # Этап 2: Фильтрация по корреляции (только для оставшихся)
    CorrFilter(threshold=0.9),
    
    # Этап 3: Тонкая фильтрация по CV (только для некоррелированных)
    CVFilter(lower_bound=0.01, upper_bound=10),
    
    # Этап 4: Финальная фильтрация outliers
    OutliersFilter(lower_percentile=0.01, upper_percentile=0.99)
])
```

**Преимущества:**
- Эффективная обработка больших данных
- Иерархическая агрегация результатов
- Последовательное уточнение анализа

#### 6. Retry паттерн — повторные попытки с разными стратегиями

Попытки выполнить анализ разными способами при неудаче.

```python
class RetryExecutor(Executor):
    def __init__(self, strategies: list[Executor], max_retries: int = 3):
        self.strategies = strategies
        self.max_retries = max_retries
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        for i, strategy in enumerate(self.strategies[:self.max_retries]):
            try:
                result = strategy.execute(data)
                if self._is_valid_result(result):
                    return result
            except Exception as e:
                if i == len(self.strategies) - 1:
                    raise e
                continue
        return data

# Использование retry паттерна
robust_matching = Experiment([
    RetryExecutor(
        strategies=[
            # Стратегия 1: Точный matching
            Experiment([
                MahalanobisDistance(),
                FaissNearestNeighbors(n_neighbors=1, faiss_mode="base")
            ]),
            
            # Стратегия 2: Приближенный matching
            Experiment([
                FaissNearestNeighbors(n_neighbors=5, faiss_mode="fast")
            ]),
            
            # Стратегия 3: Fallback на случайное сопоставление
            RandomMatching()
        ]
    ),
    MatchingQualityAssessment()
])
```

**Преимущества:**
- Устойчивость к ошибкам
- Graceful degradation
- Адаптивность к качеству данных

#### 7. Template Method паттерн — шаблонные pipeline'ы

Создание переиспользуемых шаблонов экспериментов с точками расширения.

```python
class StandardABTestTemplate(Experiment):
    """Шаблон стандартного A/B теста"""
    
    def __init__(self, 
                 custom_preprocessing: list[Executor] = None,
                 custom_tests: list[Executor] = None,
                 multitest_method: str = "bonferroni"):
        
        # Базовая предобработка
        base_preprocessing = [
            NaFiller(method="ffill"),
            OutliersFilter()
        ]
        
        # Добавляем кастомную предобработку
        preprocessing = base_preprocessing + (custom_preprocessing or [])
        
        # Базовые тесты
        base_tests = [
            GroupSizes(grouping_role=TreatmentRole()),
            GroupDifference(grouping_role=TreatmentRole()),
            TTest(grouping_role=TreatmentRole())
        ]
        
        # Добавляем кастомные тесты
        tests = base_tests + (custom_tests or [])
        
        # Собираем полный pipeline
        executors = preprocessing + tests + [
            ABAnalyzer(multitest_method=multitest_method)
        ]
        
        super().__init__(executors)

# Использование шаблона
custom_ab_test = StandardABTestTemplate(
    custom_preprocessing=[
        CategoryAggregator(threshold=10),
        DummyEncoder()
    ],
    custom_tests=[
        KSTest(grouping_role=TreatmentRole()),
        PSI(grouping_role=TreatmentRole())
    ],
    multitest_method="fdr_bh"
)
```

**Преимущества:**
- Стандартизация процессов
- Легкая кастомизация
- Соблюдение best practices

Эти паттерны можно комбинировать для создания сложных аналитических pipeline'ов, сохраняя при этом читаемость и поддерживаемость кода.

### Принципы проектирования Experiments

1. **Композируемость** — Experiments можно вкладывать друг в друга
2. **Переиспользуемость** — Общие pipeline'ы можно выделить в отдельные Experiments
3. **Конфигурируемость** — Параметры можно менять без изменения структуры
4. **Прозрачность** — Каждый шаг сохраняет свои результаты в ExperimentData
5. **Расширяемость** — Легко создавать новые типы Experiments через наследование

Experiment Framework обеспечивает мощную и гибкую систему для построения сложных аналитических pipeline'ов, сохраняя при этом простоту использования и понимания.