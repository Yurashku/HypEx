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
# | user_id | group   | pre_revenue | post_revenue |
# |---------|---------|-------------|--------------|
# | 1       | control | 100         | 110          |
# | 2       | test    | 100         | 150          |

comparator = GroupDifference(
    compare_by="columns_in_groups",
    grouping_role=TreatmentRole(),    # группировка по "group"
    baseline_role=PreTargetRole(),    # baseline: "pre_revenue"
    target_roles=TargetRole()          # сравнить с: "post_revenue"
)

# Результат:
# control: сравнение pre_revenue vs post_revenue
# test: сравнение pre_revenue vs post_revenue
```

#### Режим "cross" — перекрестное сравнение
Самый сложный режим. Сравнивает изменения между колонками в разных группах. Используется для difference-in-differences анализа.

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

GroupOperators выполняют специализированные операции над группами данных, часто используемые в matching и causal inference.

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

## 6. Analyzer'ы — комплексный анализ результатов

Analyzer'ы представляют собой специальный класс Executor'ов, которые выполняют высокоуровневый анализ результатов экспериментов. В отличие от простых вычислительных блоков (Calculator'ов), Analyzer'ы работают с результатами множества других Executor'ов, агрегируют их и принимают комплексные решения.

### Архитектура Analyzer'ов

Analyzer'ы наследуются напрямую от Executor, минуя Calculator, так как их задача — не вычисления над сырыми данными, а анализ уже полученных результатов. Типичный Analyzer:

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
    AAScoreAnalyzer()      # Выбор лучшего
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

Analyzer'ы являются ключевым компонентом для превращения множества технических результатов вычислений в понятные бизнес-решения и рекомендации.

## 7. Слой экспериментов: Experiment Framework

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
# Адаптивный выбор теста
adaptive_testing = Experiment([
    # Подготовка
    DataPreparation(),
    
    # Проверка нормальности
    NormalityTest(),
    
    # Выбор теста на основе результата
    IfExecutor(
        condition=lambda d: d.normality_test_passed,
        if_executor=TTest(),      # Параметрический тест
        else_executor=UTest()      # Непараметрический тест
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

Слой экспериментов предоставляет мощные абстракции для построения сложных аналитических pipeline'ов, сохраняя при этом простоту и читаемость кода.

## 8. Система Reporter'ов

Reporter'ы отвечают за извлечение, форматирование и представление результатов экспериментов. Они служат мостом между внутренним представлением данных в ExperimentData и форматом, удобным для пользователя или последующей обработки.

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

DictReporter является фундаментальным классом для большинства Reporter'ов в HypEx. Его философия — предоставить универсальный промежуточный формат (словарь), который легко преобразовать в любой другой.

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

Каждый наследник DictReporter переопределяет метод `report()`, комбинируя эти методы извлечения для создания нужного словаря.

### OnDictReporter — универсальный форматтер

OnDictReporter — это паттерн Decorator для DictReporter'ов. Он позволяет преобразовать базовый словарный формат в любой другой без изменения логики извлечения.

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

Каждый из этих Reporter'ов знает, какие именно результаты извлекать из ExperimentData и как их правильно интерпретировать.

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

Reporter'ы обеспечивают элегантное решение проблемы представления результатов, позволяя HypEx адаптироваться под различные use cases и требования к отчетности.