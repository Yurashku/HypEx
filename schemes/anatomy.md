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
├── OneAAStatAnalyzer — анализ
статистики
одного
A / A
теста
├── AAScoreAnalyzer — оценка
качества
A / A
тестов
и
выбор
лучшего
├── ABAnalyzer — анализ
A / B
теста
с
коррекцией
множественного
тестирования
└── MatchingAnalyzer — анализ
качества
matching
'а
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
    GroupSizes(),  # Подсчет размеров групп
    GroupDifference(),  # Вычисление разницы между группами
    TTest(),  # T-тест
    KSTest(),  # KS-тест
    ABAnalyzer()  # Анализ и коррекция множественного тестирования
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

Слой вычислений содержит конкретные реализации Calculator'ов, каждая из которых специализируется на определенном типе
операций. Все они следуют паттерну разделения вычислительной логики от работы с ExperimentData.

### Comparators: Сравнение и тестирование

Comparators отвечают за сравнение групп и проведение статистических тестов. Они имеют сложную иерархию и богатую
функциональность.

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
    target_roles=TargetRole()  # анализ "revenue" и "retention"
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
    baseline_role=PreTargetRole(),  # baseline: "pre_revenue", "pre_clicks"
    target_roles=TargetRole()  # сравнить с: "post_revenue", "post_clicks"
)

# Результат:
# Сравнение pre_revenue vs post_revenue (для всех пользователей)
# Сравнение pre_clicks vs post_clicks (для всех пользователей)
```

#### Режим "columns_in_groups" — сравнение колонок внутри каждой группы

Комбинация первых двух режимов. Сравнивает разные метрики, но отдельно для каждой группы.

```python
# Данные:
# | user_id | group   | pre_revenue | post_revenue |
# |---------|---------|-------------|--------------|
# | 1       | control | 100         | 110          |
# | 2       | test    | 100         | 150          |

comparator = GroupDifference(
    compare_by="columns_in_groups",
    grouping_role=TreatmentRole(),  # группировка по "group"
    baseline_role=PreTargetRole(),  # baseline: "pre_revenue"
    target_roles=TargetRole()  # сравнить с: "post_revenue"
)

# Результат:
# control: сравнение pre_revenue vs post_revenue
# test: сравнение pre_revenue vs post_revenue
```

#### Режим "cross" — перекрестное сравнение

Самый сложный режим. Сравнивает изменения между колонками в разных группах. Используется для difference-in-differences
анализа.

```python
comparator = GroupDifference(
    compare_by="cross",
    grouping_role=TreatmentRole(),
    baseline_role=PreTargetRole(),
    target_roles=TargetRole()
)

# Сравнивает:
# (test_post - test_pre) vs (control_post - control_pre)
```

**Применение режимов:**

- **"groups"** — классические A/B тесты, сравнение метрик между группами
- **"columns"** — анализ изменений во времени, pre/post анализ
- **"columns_in_groups"** — гетерогенные эффекты, анализ по сегментам
- **"cross"** — каузальная инференция, DiD анализ

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

Transformers изменяют сам Dataset (в отличие от других Calculator'ов, которые только вычисляют результаты). Они работают
с копией данных для обеспечения immutability.

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
        # Находим пары с высокой корреляцией
        for col1, col2 in high_corr_pairs:
            if abs(corr_matrix[col1][col2]) > threshold:
                data.roles[col2] = InfoRole()  # Понижаем роль второго
        return data
```

**OutliersFilter** — удаляет выбросы:

```python
class OutliersFilter(Transformer):
    def __init__(self,
                 lower_percentile: float = 0.05,
                 upper_percentile: float = 0.95):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def _inner_function(data: Dataset, target_cols, lower, upper) -> Dataset:
        for col in target_cols:
            q_low = data[col].quantile(lower)
            q_high = data[col].quantile(upper)
            data = data[(data[col] >= q_low) & (data[col] <= q_high)]
        return data
```

#### Обработка пропусков и категорий

**NaFiller** — заполнение пропущенных значений:

```python
class NaFiller(Transformer):
    def __init__(self, method: str = "ffill"):
        self.method = method  # 'ffill', 'bfill', 'mean', 'median', etc.

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

**SMD (Standardized Mean Difference)** — стандартизированная разница средних:

```python
class SMD(GroupOperator):
    def _inner_function(data: Dataset, test_data: Dataset) -> float:
        mean_diff = data.mean() - test_data.mean()
        pooled_std = np.sqrt((data.var() + test_data.var()) / 2)
        return mean_diff / pooled_std
```

**Bias** — оценка смещения для matching:

```python
class Bias(GroupOperator):
    def _inner_function(data: Dataset, matched_data: Dataset) -> dict:
        # Оценка качества matching через смещение
        bias = (matched_data.mean() - data.mean()) / data.std()
        return {
            "bias": bias,
            "bias_reduced": abs(bias) < 0.1  # Порог 10%
        }
```

**MatchingMetrics** — метрики качества matching:

```python
class MatchingMetrics(GroupOperator):
    def __init__(self, metric: str = "ate"):
        self.metric = metric  # 'ate', 'att', 'atc'

    def _inner_function(data: Dataset, matched_indices: Dataset) -> dict:
        if self.metric == "ate":  # Average Treatment Effect
            effect = matched_treatment.mean() - matched_control.mean()
        elif self.metric == "att":  # Average Treatment on Treated
            effect = treated.mean() - matched_control_for_treated.mean()
        # ... другие метрики
        return {"effect": effect, "se": standard_error}
```

### Splitters: Разделение данных

Splitters разделяют данные на группы для экспериментов.

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

## 6. Extension Framework

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
- Автоматически выбирают правильную реализацию для текущего backend'а

### Архитектура Extension'ов

#### Базовый класс Extension

```python
class Extension(ABC):
    """Базовый класс для всех Extension'ов в HypEx"""
    
    def __init__(self):
        # Маппинг типов backend'ов на соответствующие методы реализации
        self.BACKEND_MAPPING = {
            PandasDataset: self._calc_pandas,
            # SparkDataset: self._calc_spark,  # Для будущих backend'ов
            # DaskDataset: self._calc_dask,
        }

    @abstractmethod
    def _calc_pandas(self, data: Dataset, **kwargs):
        """Backend-специфичная реализация для pandas"""
        raise AbstractMethodError

    def calc(self, data: Dataset, **kwargs):
        """Основной метод - автоматически выбирает правильную реализацию"""
        backend_type = type(data.backend)
        implementation = self.BACKEND_MAPPING[backend_type]
        return implementation(data=data, **kwargs)

    @staticmethod
    def result_to_dataset(result: Any, roles: ABCRole | dict[str, ABCRole]) -> Dataset:
        """Утилитарный метод для преобразования результата обратно в Dataset"""
        return DatasetAdapter.to_dataset(result, roles=roles)
```

#### Принцип автоматического выбора backend'а

Extension'ы автоматически определяют тип backend'а Dataset'а и вызывают соответствующую реализацию:

```python
# Пример: Extension автоматически выбирает pandas реализацию
dataset = Dataset(data=pandas_dataframe, roles=roles)
extension = CholeskyExtension()

# calc() автоматически вызовет _calc_pandas()
result = extension.calc(dataset)
```

### Типы Extension'ов

#### 1. Базовые математические Extension'ы

Инкапсулируют фундаментальные математические операции:

```python
class CholeskyExtension(Extension):
    """Extension для разложения Холецкого"""
    
    def _calc_pandas(self, data: Dataset, epsilon: float = 1e-3, **kwargs):
        # Получаем numpy массив из pandas backend'а
        cov = data.data.to_numpy()
        
        # Добавляем регуляризацию для численной стабильности
        cov = cov + np.eye(cov.shape[0]) * epsilon
        
        # Выполняем разложение Холецкого
        cholesky_result = np.linalg.cholesky(cov)
        
        # Преобразуем результат обратно в Dataset
        return self.result_to_dataset(
            pd.DataFrame(cholesky_result, columns=data.columns),
            {column: FeatureRole() for column in data.columns}
        )

class InverseExtension(Extension):
    """Extension для обращения матрицы"""
    
    def _calc_pandas(self, data: Dataset, **kwargs):
        inverse_matrix = np.linalg.inv(data.data.to_numpy())
        
        return self.result_to_dataset(
            pd.DataFrame(inverse_matrix, columns=data.columns),
            {column: FeatureRole() for column in data.columns}
        )
```

#### 2. Encoder Extension'ы

Обеспечивают предобработку данных:

```python
class DummyEncoderExtension(Extension):
    """Extension для one-hot encoding"""
    
    @staticmethod
    def _calc_pandas(data: Dataset, target_cols: str | None = None, **kwargs):
        # Создаем dummy переменные
        dummies_df = pd.get_dummies(
            data=data[target_cols].data, 
            drop_first=True
        )
        
        # Создаем роли для новых колонок на основе исходных
        roles = {
            col: data.roles[col[:col.rfind("_")]] 
            for col in dummies_df.columns
        }
        
        # Обновляем тип данных для boolean колонок
        for role in roles.values():
            role.data_type = bool
            
        return DatasetAdapter.to_dataset(dummies_df, roles=roles)
```

#### 3. Статистические Extension'ы

Интегрируют внешние статистические библиотеки:

```python
class MultiTest(Extension):
    """Extension для коррекции множественного тестирования из statsmodels"""
    
    def __init__(self, method: ABNTestMethodsEnum, alpha: float = 0.05):
        self.method = method
        self.alpha = alpha
        super().__init__()

    def _calc_pandas(self, data: Dataset, **kwargs):
        # Извлекаем p-values из Dataset
        p_values = data.data.values.flatten()
        
        # Применяем коррекцию из statsmodels
        from statsmodels.stats.multitest import multipletests
        corrected_results = multipletests(
            p_values, 
            method=self.method.value, 
            alpha=self.alpha,
            **kwargs
        )
        
        # Формируем результат в структурированном виде
        result_data = {
            "field": [i.split(ID_SPLIT_SYMBOL)[2] for i in data.index],
            "test": [i.split(ID_SPLIT_SYMBOL)[0] for i in data.index],
            "old p-value": p_values,
            "new p-value": corrected_results[1],
            "correction": [
                j / i if j != 0 else 0.0 
                for i, j in zip(corrected_results[1], p_values)
            ],
            "rejected": corrected_results[0],
        }
        
        return DatasetAdapter.to_dataset(result_data, StatisticRole())
```

#### 4. Специализированные Extension'ы

**CompareExtension** для сравнения двух Dataset'ов:

```python
class CompareExtension(Extension, ABC):
    """Базовый класс для Extension'ов, сравнивающих два Dataset'а"""
    
    def calc(self, data: Dataset, other: Dataset | None = None, **kwargs):
        return super().calc(data=data, other=other, **kwargs)
```

**MLExtension** для машинного обучения:

```python
class MLExtension(Extension):
    """Базовый класс для ML Extension'ов с поддержкой fit/predict"""
    
    def _calc_pandas(self, data: Dataset, test_data: Dataset | None = None, 
                    mode: Literal["auto", "fit", "predict"] | None = None, **kwargs):
        
        if mode in ["auto", "fit"]:
            return self.fit(data, test_data, **kwargs)
        return self.predict(data)

    @abstractmethod
    def fit(self, X, Y=None, **kwargs):
        """Обучение модели"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X, **kwargs):
        """Предсказание"""
        raise NotImplementedError
```

### Интеграция Extension'ов с Calculator'ами

#### Правильный паттерн использования

Calculator'ы используют Extension'ы через делегирование для backend-специфичных операций:

```python
# Концептуальный пример правильного использования Extension'ов в Calculator'е
class MahalanobisDistance(Calculator):
    """Calculator для вычисления расстояния Махаланобиса"""
    
    def _inner_function(cls, data: Dataset, test_data: Dataset) -> dict:
        # Вычисление ковариационной матрицы (backend-агностично)
        control_cov = data.cov()
        test_cov = test_data.cov()
        pooled_cov = (control_cov + test_cov) / 2
        
        # Делегируем backend-специфичные операции Extension'ам
        cholesky_ext = CholeskyExtension()
        cholesky_result = cholesky_ext.calc(pooled_cov)
        
        inverse_ext = InverseExtension()
        mahalanobis_transform = inverse_ext.calc(cholesky_result)
        
        # Применяем преобразование (backend-агностично через Dataset API)
        y_control = data.dot(mahalanobis_transform.transpose())
        y_test = test_data.dot(mahalanobis_transform.transpose())
        
        return {"control": y_control, "test": y_test}
```

#### Архитектурные преимущества

**1. Автоматическая адаптация к backend'у:**
- Extension автоматически выбирает правильную реализацию
- Calculator остается backend-агностичным
- Добавление нового backend'а требует только реализации новых методов в Extension'ах

**2. Изоляция зависимостей:**
- Внешние библиотеки (numpy, scipy, sklearn) изолированы в Extension'ах
- Calculator'ы не имеют прямых зависимостей на внешние библиотеки
- Упрощается управление зависимостями и тестирование

**3. Переиспользование логики:**
- Extension'ы можно использовать в разных Calculator'ах
- Единая реализация сложных операций для всей системы

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

#### 2. Обработка результатов

Extension'ы стандартизируют возвращаемые значения через `result_to_dataset()`:

```python
def _calc_pandas(self, data: Dataset, **kwargs):
    # Выполняем backend-специфичные вычисления
    raw_result = np.some_calculation(data.data.to_numpy())
    
    # Стандартизируем результат - всегда возвращаем Dataset
    return self.result_to_dataset(
        pd.DataFrame(raw_result, columns=data.columns),
        {col: FeatureRole() for col in data.columns}
    )
```

### Создание кастомных Extension'ов

#### Шаблон для создания Extension'а

```python
class CustomExtension(Extension):
    """Шаблон для создания кастомного Extension'а"""
    
    def __init__(self, param1: float = 1.0, param2: str = "default"):
        """Инициализация с параметрами Extension'а"""
        self.param1 = param1
        self.param2 = param2
        super().__init__()
    
    def _calc_pandas(self, data: Dataset, **kwargs):
        """Реализация для pandas backend'а"""
        try:
            # Проверяем доступность необходимых библиотек
            import required_library
            
            # Извлекаем данные из Dataset
            numpy_data = data.data.to_numpy()
            
            # Выполняем backend-специфичные вычисления
            result = required_library.some_function(
                numpy_data, 
                param1=self.param1,
                param2=self.param2,
                **kwargs
            )
            
            # Преобразуем результат обратно в Dataset
            if isinstance(result, np.ndarray):
                result_df = pd.DataFrame(result, columns=data.columns)
                roles = {col: FeatureRole() for col in data.columns}
            else:
                # Обработка других типов результатов
                result_df = pd.DataFrame({"result": [result]})
                roles = {"result": StatisticRole()}
            
            return self.result_to_dataset(result_df, roles)
            
        except ImportError:
            raise RuntimeError(
                "CustomExtension требует библиотеку 'required_library'. "
                "Установите: pip install required_library"
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка в CustomExtension: {e}")
    
    # Для поддержки других backend'ов добавьте соответствующие методы:
    # def _calc_spark(self, data: Dataset, **kwargs):
    #     """Реализация для Spark backend'а"""
    #     pass
```

### Архитектурные принципы Extension'ов

#### 1. Принцип единого интерфейса

Все Extension'ы предоставляют метод `calc()` с одинаковой сигнатурой:

```python
# Единый интерфейс для всех Extension'ов
result = extension.calc(data=dataset, **optional_params)
```

#### 2. Принцип изоляции зависимостей

Extension'ы изолируют внешние зависимости от основной логики:

```python
# Правильно: зависимости изолированы в Extension
class ScipyStatsExtension(Extension):
    def _calc_pandas(self, data: Dataset, **kwargs):
        from scipy.stats import ttest_ind  # Изолированный import
        # ... использование scipy
        
# Неправильно: прямое использование в Calculator
class BadCalculator(Calculator):
    def _inner_function(data: Dataset, **kwargs):
        from scipy.stats import ttest_ind  # Нарушение архитектуры!
        # ... это нарушает backend-агностичность
```

#### 3. Принцип автоматического выбора реализации

Extension'ы автоматически выбирают оптимальную реализацию:

```python
class AdaptiveExtension(Extension):
    def __init__(self):
        super().__init__()
        # Расширяем mapping при добавлении новых backend'ов
        if hasattr(self, '_calc_spark'):
            self.BACKEND_MAPPING[SparkDataset] = self._calc_spark
        if hasattr(self, '_calc_dask'):
            self.BACKEND_MAPPING[DaskDataset] = self._calc_dask
```

#### 4. Принцип graceful degradation

Extension'ы обеспечивают работу даже при частичной недоступности функций:

```python
class RobustExtension(Extension):
    def _calc_pandas(self, data: Dataset, **kwargs):
        try:
            # Пытаемся использовать оптимизированную реализацию
            import scipy.linalg
            return self._fast_implementation(data, **kwargs)
        except ImportError:
            # Fallback на базовую реализацию
            return self._basic_implementation(data, **kwargs)
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

Выполняет эксперимент заданное количество раз:

```python
class CycledExperiment(ExperimentWithReporter):
    def __init__(self,
                 executors: Sequence[Executor],
                 reporter: DatasetReporter,
                 n_iterations: int = 10):
        super().__init__(executors, reporter)
        self.n_iterations = n_iterations

    def execute(self, data: ExperimentData) -> ExperimentData:
        results = []
        for i in range(self.n_iterations):
            # Каждая итерация начинается с чистых данных
            iteration_result = self.one_iteration(data, key=str(i))
            results.append(iteration_result)

        # Объединяем результаты всех итераций
        combined_results = Dataset.concat(results)

        return data.set_value(
            space=ExperimentDataEnum.analysis_tables,
            executor_id=self.id,
            value=combined_results
        )
```

**Применение:**

- Bootstrap анализ
- Оценка стабильности результатов
- Monte Carlo симуляции

### GroupExperiment — выполнение по группам

Применяет эксперимент к каждой группе данных отдельно:

```python
class GroupExperiment(ExperimentWithReporter):
    def __init__(self,
                 executors: Sequence[Executor],
                 reporter: DatasetReporter,
                 searching_role: ABCRole = GroupingRole()):
        super().__init__(executors, reporter)
        self.searching_role = searching_role

    def execute(self, data: ExperimentData) -> ExperimentData:
        # Находим поле для группировки
        group_field = data.ds.search_columns([self.searching_role])[0]

        results = []
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
            results.append(iteration_result)

        return self._set_result(data, results)
```

### IfParamsExperiment — параметрический поиск с условием остановки

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

#### 7. Observer паттерн — мониторинг выполнения

Добавление логирования и мониторинга в процесс выполнения.

```python
class ObservableExperiment(Experiment):
    def __init__(self, executors: list[Executor], observers: list[Observer] = None):
        super().__init__(executors)
        self.observers = observers or []

    def execute(self, data: ExperimentData) -> ExperimentData:
        for i, executor in enumerate(self.executors):
            # Уведомляем о начале
            for observer in self.observers:
                observer.on_executor_start(executor, data)

            # Выполнение
            start_time = time.time()
            data = executor.execute(data)
            execution_time = time.time() - start_time

            # Уведомляем о завершении
            for observer in self.observers:
                observer.on_executor_complete(executor, data, execution_time)

        return data


# Использование
experiment = ObservableExperiment(
    executors=[...],
    observers=[
        LoggingObserver(),
        MetricsCollector(),
        ProgressBar(),
        AlertingObserver(threshold=60)  # Алерт если executor > 60 сек
    ]
)
```

**Преимущества:**

- Прозрачность выполнения
- Сбор метрик и диагностика
- Возможность прерывания при проблемах

### Лучшие практики при проектировании экспериментов

1. **Модульность** — разбивайте сложные эксперименты на логические блоки
2. **Переиспользование** — создавайте библиотеку стандартных подэкспериментов
3. **Валидация** — добавляйте проверки между этапами
4. **Документирование** — используйте говорящие имена и комментарии
5. **Тестирование** — тестируйте эксперименты на небольших данных
6. **Версионирование** — сохраняйте версии успешных экспериментов
7. **Мониторинг** — логируйте ключевые метрики выполнения

Слой экспериментов предоставляет мощные абстракции для построения сложных аналитических pipeline'ов, сохраняя при этом
простоту и читаемость кода.

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

#### Стандартные форматтеры

**DatasetReporter** — табличное представление:

```python
class DatasetReporter(OnDictReporter):
    def report(self, data: ExperimentData):
        # Получаем базовый dict
        dict_report = self.dict_reporter.report(data)
        # Преобразуем в структурированную таблицу
        return self.convert_flat_dataset(dict_report)
```

**Потенциальные форматтеры (roadmap):**

**HTMLReporter** — интерактивные HTML отчеты:

- Таблицы с сортировкой и фильтрацией
- Графики и визуализации
- Collapsible секции для детализации
- Экспорт в различные форматы

**PDFReporter** — профессиональные PDF отчеты:

- Форматированные таблицы и графики
- Executive summary на первой странице
- Детальные приложения с методологией
- Брендирование и стилизация

**MarkdownReporter** — отчеты для документации:

- Структурированный markdown
- Таблицы в GFM формате
- Встроенные графики как base64
- Готов для вставки в wiki/confluence

**JSONReporter** — машиночитаемый формат:

- Полная сериализация результатов
- Метаданные об эксперименте
- Версионирование схемы
- Поддержка streaming

**ExcelReporter** — multi-sheet Excel файлы:

- Summary на первом листе
- Детальные результаты по листам
- Условное форматирование
- Встроенные формулы и графики

#### Создание кастомного форматтера

```python
class CustomFormatter(OnDictReporter):
    def __init__(self, dict_reporter: DictReporter, format_options: dict):
        super().__init__(dict_reporter)
        self.format_options = format_options

    def report(self, data: ExperimentData):
        # Получаем базовый словарь
        base_dict = self.dict_reporter.report(data)

        # Применяем кастомное форматирование
        formatted = self.apply_formatting(base_dict)

        # Добавляем метаданные
        formatted['metadata'] = self.extract_metadata(data)

        # Возвращаем в нужном формате
        return self.render(formatted)
```

### Использование Reporter'ов в Experiment

Reporter'ы интегрированы в систему экспериментов через класс ExperimentWithReporter.

#### ExperimentWithReporter

Этот класс добавляет автоматическую генерацию отчетов к экспериментам:

```python
class ExperimentWithReporter(Experiment):
    def __init__(self, executors: list, reporter: Reporter):
        super().__init__(executors)
        self.reporter = reporter

    def one_iteration(self, data: ExperimentData, key: str = ""):
        # Выполняем эксперимент
        result_data = super().execute(data)
        # Автоматически генерируем отчет
        return self.reporter.report(result_data)
```

#### Специализированные эксперименты с Reporter'ами

**ParamsExperiment** — всегда требует Reporter:

- После каждой комбинации параметров генерирует отчет
- Агрегирует отчеты всех итераций
- Позволяет сравнить результаты разных параметров

**GroupExperiment** — Reporter для каждой группы:

- Генерирует отчет для каждой группы отдельно
- Объединяет в общую таблицу с индексом по группам

**CycledExperiment** — Reporter для каждого цикла:

- Отчет после каждой итерации
- Статистика по всем итерациям

#### Паттерны использования

**Inline reporter в эксперименте:**

```python
experiment = ParamsExperiment(
    executors=[...],
    params={...},
    reporter=DatasetReporter(ABDictReporter())
)
```

**Композиция репортеров:**

```python
base_reporter = TestDictReporter()
formatted_reporter = DatasetReporter(base_reporter)

experiment = ExperimentWithReporter(
    executors=[...],
    reporter=formatted_reporter
)
```

**Множественные отчеты:**

```python
data = experiment.execute(initial_data)

# Разные форматы из одних данных
summary = SummaryDictReporter().report(data)
detailed = DetailedDatasetReporter().report(data)
visual = VisualizationReporter().report(data)
```

### Иерархия конкретных Reporter'ов

HypEx предоставляет набор готовых Reporter'ов для типовых задач:

#### Для статистических тестов

- **TestDictReporter** — базовый класс для тестовых репортеров
- **OneAADictReporter** — отчет по одному A/A тесту
- **AADatasetReporter** — табличный отчет по A/A тестам
- **ABDictReporter** — отчет по A/B тесту
- **ABDatasetReporter** — табличный отчет по A/B тесту
- **HomoDictReporter** — отчет по тесту гомогенности
- **HomoDatasetReporter** — табличный отчет по гомогенности

#### Для matching

- **MatchingDictReporter** — базовый отчет по matching
- **MatchingDatasetReporter** — табличный отчет по matching
- **MatchingQualityDictReporter** — отчет по качеству matching
- **MatchingQualityDatasetReporter** — табличный отчет по качеству

#### Специальные репортеры

- **AAPassedReporter** — определение прошедших A/A тестов
- **AABestSplitReporter** — отчет о лучшем разбиении

Каждый из этих Reporter'ов знает, какие именно результаты извлекать из ExperimentData и как их правильно
интерпретировать.

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

Shell слой реализует ключевой принцип библиотеки HypEx: 90% практических задач должны решаться в 2 строчки кода. Каждый
Shell создавался на основе анализа реальных индустриальных потребностей:

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

- Базовый A/B тест + дополнительные методы проверки качества
- Нестандартная комбинация стандартных блоков
- Требуется кастомизация параметров анализа
- Команда готова работать с более сложным API

**Решение — композиция Executor'ов:**

```python
from hypex.experiments.base import Experiment, OnRoleExperiment
from hypex.comparators import TTest, KSTest, PSI
from hypex.operators import GroupSizes, GroupDifference
from hypex.analyzers.ab import ABAnalyzer
from hypex.dataset import TargetRole

# Создание кастомного эксперимента
custom_ab_experiment = Experiment([
    # Стандартная часть
    OnRoleExperiment([
        GroupSizes(grouping_role=TreatmentRole()),
        GroupDifference(grouping_role=TreatmentRole()),
        TTest(grouping_role=TreatmentRole()),
        KSTest(grouping_role=TreatmentRole()),

        # Дополнительный анализ стабильности
        PSI(grouping_role=TreatmentRole()),  # Стабильность распределения

    ], role=TargetRole()),

    # Анализ результатов с менее консервативной коррекцией
    ABAnalyzer(multitest_method="fdr_bh")  # False Discovery Rate вместо Bonferroni
])

# Выполнение
experiment_data = ExperimentData(dataset)
results = custom_ab_experiment.execute(experiment_data)

# Извлечение результатов через Reporter
from hypex.reporters.ab import ABDictReporter

reporter = ABDictReporter()
formatted_results = reporter.report(results)

print("=== РАСШИРЕННЫЙ A/B АНАЛИЗ ===")
print(f"T-test: {formatted_results['ttest']}")
print(f"KS-test: {formatted_results['kstest']}")
print(f"PSI (стабильность групп): {formatted_results['psi']}")
print(f"FDR-скорректированные результаты: {formatted_results['multitest']}")
```

**Что изменилось:**

- Перешли от Shell к прямой композиции Executor'ов
- Добавили PSI для проверки качества разбиения на группы
- Изменили метод коррекции множественного тестирования на менее консервативный
- Используем Reporter для извлечения результатов

**Результат:**

```
T-test: p-value = 0.032, effect = 1.6%
KS-test: p-value = 0.045, distribution differs
PSI = 0.08 (группы сбалансированы, PSI < 0.1)
FDR-скорректированные p-values: [0.038, 0.048]
Рекомендация: Эффект значим, группы качественно сбалансированы
```

**Почему потребовался переход на уровень композиции:**

- Нужен PSI анализ, который не включен в стандартный ABTest
- Требуется другой метод коррекции множественного тестирования
- Кастомизация набора выполняемых тестов

### Сценарий 3: Кастомный анализ для специфической метрики (Уровень 6 — Расширение)

**Бизнес-задача:** Продуктовая команда анализирует влияние изменений на конверсию в покупку, которая рассчитывается как
отношение (ratio): покупки / уникальные пользователи. Стандартные тесты не учитывают специфику ratio метрик.

**Характеристики задачи:**

- Специфический тип метрики (ratio: события/пользователи)
- Нужен специализированный статистический тест с delta method
- Стандартные Calculator'ы не покрывают потребность
- Команда имеет статистическую экспертизу

**Решение — создание кастомного Calculator'а:**

```python
from hypex.executors.calculator import Comparator
from hypex.dataset import Dataset
import numpy as np
from scipy import stats


class RatioTest(Comparator):
    """
    Статистический тест для сравнения ratio метрик между группами.
    Использует delta method для корректной оценки стандартной ошибки.
    """

    def __init__(self, alpha: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _inner_function(self, control_data: Dataset, test_data: Dataset) -> dict:
        """
        Выполняет тест для ratio метрик.
        
        Ожидает колонки:
        - events: количество событий (покупок)
        - users: количество уникальных пользователей
        """

        # Извлечение данных
        control_events = control_data['events'].sum()
        control_users = control_data['users'].sum()
        test_events = test_data['events'].sum()
        test_users = test_data['users'].sum()

        # Расчет ratio метрик
        control_ratio = control_events / control_users if control_users > 0 else 0
        test_ratio = test_events / test_users if test_users > 0 else 0

        # Delta method для стандартной ошибки ratio
        control_se = self._delta_method_se(control_events, control_users)
        test_se = self._delta_method_se(test_events, test_users)

        # Pooled standard error для разности
        pooled_se = np.sqrt(control_se ** 2 + test_se ** 2)

        # Z-test для разности ratio
        if pooled_se > 0:
            z_stat = (test_ratio - control_ratio) / pooled_se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0
            p_value = 1.0

        # Доверительный интервал для разности
        diff = test_ratio - control_ratio
        ci_margin = stats.norm.ppf(1 - self.alpha / 2) * pooled_se
        ci_lower = diff - ci_margin
        ci_upper = diff + ci_margin

        return {
            "control_ratio": control_ratio,
            "test_ratio": test_ratio,
            "ratio_difference": diff,
            "relative_lift": (diff / control_ratio * 100) if control_ratio > 0 else 0,
            "statistic": z_stat,
            "p-value": p_value,
            "pass": p_value < self.alpha,
            "confidence_interval": [ci_lower, ci_upper],
            "interpretation": self._interpret_results(p_value, diff, control_ratio)
        }

    def _delta_method_se(self, events: float, users: float) -> float:
        """Вычисляет стандартную ошибку ratio через delta method"""
        if users <= 0:
            return 0

        ratio = events / users
        # Delta method: SE(X/Y) ≈ sqrt(Var(X)/Y² + X²*Var(Y)/Y⁴ - 2*X*Cov(X,Y)/Y³)
        # Для Poisson процесса: Var(events) = events, Cov(events, users) ≈ 0
        variance = (events / users ** 2) + (ratio ** 2 * users / users ** 2)  # Упрощенная формула
        return np.sqrt(variance / users)  # Нормализация на размер выборки

    def _interpret_results(self, p_value: float, diff: float, control_ratio: float) -> str:
        """Генерирует текстовую интерпретацию результатов"""
        if p_value >= self.alpha:
            return "Нет статистически значимой разницы в конверсии"

        if diff > 0:
            lift_pct = (diff / control_ratio * 100) if control_ratio > 0 else 0
            return f"Конверсия выше в тестовой группе на {lift_pct:.1f}% (улучшение)"
        else:
            drop_pct = abs(diff / control_ratio * 100) if control_ratio > 0 else 0
            return f"Конверсия ниже в тестовой группе на {drop_pct:.1f}% (ухудшение)"


# Создание эксперимента с кастомным тестом
ratio_experiment = Experiment([
    OnRoleExperiment([
        GroupSizes(grouping_role=TreatmentRole()),

        # Кастомный анализ ratio метрик
        RatioTest(
            grouping_role=TreatmentRole(),
            alpha=0.05
        ),

        # Дополнительные стандартные тесты для validation
        TTest(grouping_role=TreatmentRole()),  # Для проверки consistency

    ], role=TargetRole()),
])

# Подготовка данных для ratio анализа
ratio_dataset = Dataset(
    data=experiment_data,
    roles={
        'user_group': TreatmentRole(),
        'conversion_events': TargetRole(),  # Количество покупок
        'unique_users': TargetRole(),  # Количество уникальных пользователей
        'revenue_per_user': TargetRole()  # Дополнительная метрика для t-test
    }
)

# Выполнение анализа
experiment_data = ExperimentData(ratio_dataset)
results = ratio_experiment.execute(experiment_data)
```

**Результат кастомного анализа:**

```
=== RATIO METRICS АНАЛИЗ ===
Control группа: 1247 покупок / 4520 пользователей = 27.6% конверсия
Test группа: 1456 покупок / 4580 пользователей = 31.8% конверсия
Difference: +4.2 п.п. (relative lift: +15.2%)
Z-statistic: 3.24, p-value = 0.001 (высоко значимо)
95% CI для разности: [1.7%, 6.7%]
Интерпретация: Конверсия выше в тестовой группе на 15.2% (улучшение)

=== VALIDATION ===
T-test на revenue per user: p-value = 0.018 (согласуется с ratio тестом)
```

**Почему потребовался уровень расширения:**

- Ratio метрики требуют специального подхода (delta method для SE)
- Стандартные тесты дают некорректные результаты для отношений
- Нужна доменная экспертиза для правильной статистической модели
- Требуется кастомная интерпретация результатов (relative lift)

**Преимущества архитектурного подхода:**

- Новый Calculator легко интегрируется в существующие Experiment'ы
- Следует всем конвенциям библиотеки (единый интерфейс, ExperimentData)
- Можно комбинировать с стандартными компонентами для validation
- Переиспользуется в других экспериментах с ratio метриками

### Сценарий 4: Сложный многоэтапный matching анализ (Комбинация уровней)

**Бизнес-задача:** Аналитическая команда исследует эффект маркетинговой кампании на retention пользователей. Данные
observational (без рандомизации), поэтому нужен sophisticated matching анализ с проверкой качества и sensitivity
анализом.

**Характеристики задачи:**

- Многоэтапный процесс: подготовка → matching → валидация → анализ
- Комбинация готовых решений и кастомных компонентов
- Требования к quality assurance на каждом этапе
- Нужна детальная диагностика и отчетность

**Решение — комбинация Shell + Композиция + Расширение:**

```python
# Этап 1: Быстрая оценка с помощью Shell (базовый matching)
initial_matching = Matching(
    distance="mahalanobis",
    bias_estimation=True,
    quality_tests=["ttest", "kstest"]
)

initial_results = initial_matching.execute(dataset)
print(f"Базовое качество matching: {initial_results.quality_results}")

# Этап 2: Если качество неудовлетворительное - детальная настройка
if initial_results.quality_results['overall_quality'] < 0.8:

    # Кастомный preprocessor для улучшения matching
    class AdvancedPreprocessor(Transformer):
        """Продвинутая предобработка для улучшения quality matching'а"""

        def _inner_function(self, data: Dataset) -> Dataset:
            # Логарифмирование скошенных переменных
            log_features = ['income', 'session_duration']
            for feature in log_features:
                data[f'{feature}_log'] = np.log1p(data[feature])

            # Создание interaction features
            data['age_income_interaction'] = data['age'] * data['income']

            # Binning категориальных переменных с малыми группами
            rare_categories = data['device_type'].value_counts()
            data['device_type_grouped'] = data['device_type'].map(
                lambda x: x if rare_categories[x] > 50 else 'other'
            )

            return data


    # Кастомный качественный анализ
    class SensitivityAnalyzer(Executor):
        """Sensitivity анализ для оценки робастности результатов"""

        def execute(self, data: ExperimentData) -> ExperimentData:
            matched_data = data.additional_fields['matched_pairs']

            # Анализ с разными distance thresholds
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
            sensitivity_results = []

            for threshold in thresholds:
                # Фильтруем пары по качеству matching
                high_quality_pairs = matched_data[
                    matched_data['match_distance'] <= threshold
                    ]

                # Вычисляем эффект на filtered данных
                effect = self._calculate_treatment_effect(high_quality_pairs)

                sensitivity_results.append({
                    'threshold': threshold,
                    'n_pairs': len(high_quality_pairs),
                    'treatment_effect': effect,
                    'coverage': len(high_quality_pairs) / len(matched_data)
                })

            # Сохраняем результаты sensitivity анализа
            sensitivity_dataset = Dataset(data=pd.DataFrame(sensitivity_results))
            return data.set_value(
                space=ExperimentDataEnum.analysis_tables,
                executor_id=self.id,
                value=sensitivity_dataset
            )


    # Композиция продвинутого matching pipeline
    advanced_matching = Experiment([
        # Этап 1: Продвинутая предобработка
        AdvancedPreprocessor(),
        OutliersFilter(method="isolation_forest"),
        NaFiller(method="knn"),

        # Этап 2: Matching с несколькими алгоритмами
        MahalanobisDistance(grouping_role=TreatmentRole()),
        FaissNearestNeighbors(n_neighbors=3),  # Больше кандидатов

        # Этап 3: Выбор лучших пар
        MatchingQualityFilter(min_quality_score=0.8),

        # Этап 4: Анализ качества
        Bias(grouping_role=TreatmentRole()),
        SMD(grouping_role=TreatmentRole()),  # Standardized Mean Difference

        # Этап 5: Основной анализ
        OnRoleExperiment([
            TTest(grouping_role=TreatmentRole()),
            KSTest(grouping_role=TreatmentRole()),
        ], role=TargetRole()),

        # Этап 6: Sensitivity анализ
        SensitivityAnalyzer(),

        # Этап 7: Итоговый анализ
        MatchingAnalyzer()
    ])

    # Выполнение продвинутого анализа
    final_results = advanced_matching.execute(ExperimentData(dataset))


# Этап 3: Детальная отчетность с кастомным Reporter
class ComprehensiveMatchingReporter(DictReporter):
    """Подробный отчет по всем этапам matching анализа"""

    def report(self, data: ExperimentData) -> dict:
        base_results = super().report(data)

        # Добавляем sensitivity анализ
        sensitivity_data = data.analysis_tables.get('SensitivityAnalyzer╤╤')
        if sensitivity_data is not None:
            base_results['sensitivity_analysis'] = {
                'stability_assessment': self._assess_stability(sensitivity_data),
                'robust_effect_estimate': self._robust_estimate(sensitivity_data),
                'recommended_threshold': self._recommend_threshold(sensitivity_data)
            }

        # Добавляем качественную оценку
        base_results['quality_assessment'] = self._comprehensive_quality_check(data)

        return base_results


comprehensive_reporter = ComprehensiveMatchingReporter()
final_report = comprehensive_reporter.report(final_results)
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

### Паттерны выбора подхода

#### Матрица принятия решений

| Критерий                      | Shell (Уровень 4)            | Композиция (Уровень 5)   | Расширение (Уровень 6) |
|-------------------------------|------------------------------|--------------------------|------------------------|
| **Стандартность задачи**      | Типовая (A/B, A/A, Matching) | Нестандартная комбинация | Уникальная методология |
| **Экспертиза команды**        | Базовая                      | Средняя                  | Высокая                |
| **Временные рамки**           | Срочно (часы)                | Умеренно (дни)           | Не критично (недели)   |
| **Требования к кастомизации** | Минимальные                  | Средние                  | Максимальные           |
| **Частота использования**     | Разовая или регулярная       | Регулярная               | Проектная              |

#### Типичные сигналы для перехода на следующий уровень

**Shell → Композиция:**

- "Нужно добавить еще один тест"
- "Хотим изменить порядок выполнения"
- "Требуется нестандартная комбинация методов"
- "Shell делает не совсем то, что нужно"

**Композиция → Расширение:**

- "Нужного Executor'а нет в библиотеке"
- "Требуется интеграция с внешней библиотекой"
- "Нужна доменно-специфическая логика"
- "Стандартные методы не подходят для наших данных"

#### Антипаттерны и их решения

**Антипаттерн 1: Преждевременная оптимизация**

```python
# Плохо: сразу создавать сложный custom Executor
class ComplexCustomAnalyzer(Executor):


# 200 строк сложной логики для простой задачи

# Хорошо: начать с Shell и усложнять по необходимости
ab_test = ABTest()  # Сначала проверить, что базовое решение не подходит
```

**Антипаттерн 2: Игнорирование стандартных решений**

```python
# Плохо: переизобретать велосипед
custom_experiment = Experiment([
    ManualGroupSplit(),  # Вместо стандартного AASplitter
    ManualTTest(),  # Вместо стандартного TTest
    ManualReporting()  # Вместо стандартных Reporter'ов
])

# Хорошо: использовать стандартные блоки где возможно
experiment = Experiment([
    AASplitter(),  # Стандартный, надежный, протестированный
    OnRoleExperiment([TTest()], role=TargetRole()),
    CustomSpecificAnalyzer()  # Только то, что действительно уникально
])
```

**Антипаттерн 3: Монолитные кастомные решения**

```python
# Плохо: один большой Executor делает все
class MonolithicAnalyzer(Executor):
    def execute(self, data):
# Предобработка + анализ + отчетность в одном классе
# Трудно тестировать, переиспользовать, поддерживать


# Хорошо: разбить на композируемые части
experiment = Experiment([
    CustomPreprocessor(),  # Одна ответственность
    StandardAnalysis(),  # Переиспользуемый компонент
    CustomReporter()  # Отдельный форматтер
])
```

### Выводы и рекомендации

**Принцип прогрессивного усложнения:**
Всегда начинайте с самого простого решения, которое решает задачу. HypEx позволяет органично развивать решение по мере
роста требований.

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

