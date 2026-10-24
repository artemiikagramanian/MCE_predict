#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os

# Запрещаем библиотекам создавать много потоков
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

print("Количество потоков ограничено", flush=True)


# In[27]:


import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pymatgen.core import Composition
from xenonpy.datatools import preset

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning
)

print("Python:", sys.executable)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Импорты выполнены", flush=True)


# In[29]:


INPUT_FILE = Path(
    r"/Users/artemykagramanian/Desktop/ампермагнит/MCE + ML/итоговые датасеты/DB_mc.xlsx"
)

INPUT_SHEET = "1"

OUTPUT_DIRECTORY = Path(
    r"/Users/artemykagramanian/Desktop/ампермагнит/MCE + ML/обработанные датасеты"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "xenonpy_checkpoints"
)

CHECKPOINT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_XLSX = (
    OUTPUT_DIRECTORY
    / "RCP_5T_XenonPy_descriptors.xlsx"
)

OUTPUT_CSV = (
    OUTPUT_DIRECTORY
    / "RCP_5T_XenonPy_ML_data.csv"
)

print("Входной файл:", INPUT_FILE)
print("Папка результатов:", OUTPUT_DIRECTORY)
print("Пути подготовлены", flush=True)


# In[31]:


df = pd.read_excel(
    INPUT_FILE,
    sheet_name=INPUT_SHEET
)

required_columns = [
    "Compound",
    "Property",
    "Value",
    "Magnetic Field",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Отсутствуют столбцы: {missing_columns}"
    )

df["Compound"] = (
    df["Compound"]
    .astype("string")
    .str.strip()
)

df["Property"] = (
    df["Property"]
    .astype("string")
    .str.strip()
)

df["Value"] = pd.to_numeric(
    df["Value"],
    errors="coerce"
)

df["Magnetic Field"] = pd.to_numeric(
    df["Magnetic Field"],
    errors="coerce"
)

if "D_crystalite [nm]" in df.columns:
    df["D_crystalite [nm]"] = pd.to_numeric(
        df["D_crystalite [nm]"],
        errors="coerce"
    )

df = df.dropna(
    subset=[
        "Compound",
        "Property",
        "Value",
    ]
).copy()

df = df.reset_index(
    drop=True
)

print("Размер таблицы:", df.shape)
print(df.head())
print("Таблица загружена", flush=True)


# In[32]:


me_data = df.loc[
    df["Property"].eq("RCP (J/Kg)")
].copy()

me_group_sizes = (
    me_data.groupby("Magnetic Field", dropna=True)
    .size()
    .sort_values(ascending=False)
)

print("Размеры групп ME:")
print(me_group_sizes)

TARGET_FIELD = 5.0  # нужное поле, Тл

available_fields = sorted(me_data["Magnetic Field"].dropna().unique())
if TARGET_FIELD not in available_fields:
    raise ValueError(
        f"Поля {TARGET_FIELD} Т нет в данных. Доступные значения: {available_fields}"
    )

largest_me_field = TARGET_FIELD  # имя переменной не меняю специально —
                                  # весь код ниже (заголовки графиков, принты) на неё ссылается

largest_me_group = me_data.loc[
    me_data["Magnetic Field"].eq(largest_me_field)
].copy()

largest_me_group = largest_me_group.dropna(
    subset=["Value"]
).copy()

largest_me_group = largest_me_group.reset_index(
    names="Original_index"
)

largest_me_group["delta_S_M"] = (
    largest_me_group["Value"]
    .astype(float)
)

largest_me_group["log_delta_S_M"] = np.log1p(
    largest_me_group["delta_S_M"]
)

print(
    f"\nВыбрано поле: {largest_me_field:g} Т"
)

print(
    f"Количество измерений: {len(largest_me_group)}"
)

print(
    f"Уникальных записей Compound: "
    f"{largest_me_group['Compound'].nunique()}"
)

print("Группа ME подготовлена", flush=True)


# In[33]:


print("Начало обработки выбросов", flush=True)

required_columns = [
    "delta_S_M",
]

missing_columns = [
    column
    for column in required_columns
    if column not in largest_me_group.columns
]

if missing_columns:
    raise ValueError(
        f"В largest_me_group отсутствуют столбцы: "
        f"{missing_columns}"
    )

print(
    "Столбец delta_S_M найден",
    flush=True
)

# Принудительно преобразуем данные в обычный NumPy-массив float64
me_values_array = pd.to_numeric(
    largest_me_group["delta_S_M"],
    errors="coerce"
).to_numpy(
    dtype=np.float64
)

print(
    "Преобразование в NumPy выполнено",
    flush=True
)

print(
    "Размер массива:",
    me_values_array.size,
    flush=True
)

# Оставляем только конечные числовые значения
finite_mask = np.isfinite(
    me_values_array
)

me_values_array = me_values_array[
    finite_mask
]

print(
    "Числовых конечных значений:",
    me_values_array.size,
    flush=True
)

if me_values_array.size == 0:
    raise ValueError(
        "После очистки не осталось числовых значений delta_S_M"
    )

print(
    "Минимум:",
    np.min(me_values_array),
    flush=True
)

print(
    "Максимум:",
    np.max(me_values_array),
    flush=True
)


# In[37]:


print(
    "Начало расчёта квартилей",
    flush=True
)

Q1 = float(
    np.percentile(
        me_values_array,
        25
    )
)

Q3 = float(
    np.percentile(
        me_values_array,
        75
    )
)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print(
    "Квартили рассчитаны",
    flush=True
)

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Нижняя граница:", lower_limit)
print("Верхняя граница:", upper_limit)

# Работаем через отдельный NumPy-массив
delta_s_values = pd.to_numeric(
    largest_me_group["delta_S_M"],
    errors="coerce"
).to_numpy(
    dtype=np.float64
)

target_outlier_mask = (
    ~np.isfinite(delta_s_values)
    | (delta_s_values < lower_limit)
    | (delta_s_values > upper_limit)
    | (delta_s_values <= 0)
)

print(
    "Маска выбросов рассчитана",
    flush=True
)

largest_me_group = (
    largest_me_group
    .copy()
)

largest_me_group.loc[
    :,
    "Is_target_outlier"
] = target_outlier_mask

values_for_histogram = delta_s_values[
    ~target_outlier_mask
]

print(
    "Количество исходных значений:",
    len(delta_s_values),
    flush=True
)

print(
    "Количество выбросов:",
    int(np.sum(target_outlier_mask)),
    flush=True
)

print(
    "Количество значений для гистограммы:",
    len(values_for_histogram),
    flush=True
)

print(
    "Расчёт выбросов успешно завершён",
    flush=True
)


# In[39]:


# ==================================================================
# Безопасная гистограмма через Plotly
# ==================================================================

print(
    "Начало построения гистограммы Plotly",
    flush=True
)

import plotly.graph_objects as go


# Преобразуем данные в NumPy-массив
values_for_histogram = np.asarray(
    values_for_histogram,
    dtype=np.float64
)

# Оставляем только конечные положительные значения
values_for_histogram = values_for_histogram[
    np.isfinite(values_for_histogram)
    & (values_for_histogram > 0)
]

if values_for_histogram.size == 0:
    raise ValueError(
        "Нет положительных значений для гистограммы"
    )

print(
    "Количество значений:",
    len(values_for_histogram),
    flush=True
)

print(
    "Минимум:",
    values_for_histogram.min(),
    flush=True
)

print(
    "Максимум:",
    values_for_histogram.max(),
    flush=True
)


# Логарифмируем значения
log_values = np.log10(
    values_for_histogram
)

print(
    "Логарифмирование выполнено",
    flush=True
)


# Создаём интерактивную гистограмму
figure = go.Figure()

figure.add_trace(
    go.Histogram(
        x=log_values,
        nbinsx=20,
        marker={
            "color": "#2878B5",
            "line": {
                "color": "white",
                "width": 1,
            },
        },
        name="Количество наблюдений",
    )
)

figure.update_layout(
    title=(
        f"Распределение изменения магнитной энтропии "
        f"при {largest_me_field:g} Т<br>"
        f"без выбросов, n={len(values_for_histogram)}"
    ),
    xaxis_title=(
        "log₁₀(ΔSₘ), "
        "ΔSₘ в Дж/(кг·К)"
    ),
    yaxis_title="Количество наблюдений",
    template="plotly_white",
    width=1000,
    height=600,
    bargap=0.05,
)

figure.update_xaxes(
    showgrid=True,
    gridcolor="rgba(0, 0, 0, 0.1)"
)

figure.update_yaxes(
    showgrid=True,
    gridcolor="rgba(0, 0, 0, 0.1)"
)


# Сохраняем график в HTML
histogram_html_file = (
    OUTPUT_DIRECTORY
    / "ME_log10_histogram_without_outliers.html"
)

figure.write_html(
    histogram_html_file,
    include_plotlyjs=True,
    full_html=True,
    auto_open=False
)

print(
    "Интерактивная гистограмма сохранена:",
    histogram_html_file.resolve(),
    flush=True
)


# In[41]:


def clean_formula(formula):
    if pd.isna(formula):
        return np.nan

    return (
        str(formula)
        .strip()
        .replace(" ", "")
        .replace("−", "-")
        .replace("–", "-")
        .replace(",", ".")
    )


def formula_to_dict(formula):
    try:
        return Composition(
            formula
        ).get_el_amt_dict()

    except Exception as error:
        print(
            f"Ошибка формулы {formula!r}: {error}",
            flush=True
        )

        return None


largest_me_group["Clean_formula"] = (
    largest_me_group["Compound"]
    .apply(clean_formula)
)

largest_me_group["composition"] = (
    largest_me_group["Clean_formula"]
    .apply(formula_to_dict)
)

failed_formulas = largest_me_group.loc[
    largest_me_group["composition"].isna()
].copy()

descriptor_source = largest_me_group.loc[
    largest_me_group["composition"].notna()
].copy()

unique_compositions = (
    descriptor_source[
        [
            "Clean_formula",
            "composition",
        ]
    ]
    .drop_duplicates(
        subset="Clean_formula"
    )
    .reset_index(drop=True)
)

print(
    "Успешных строк:",
    len(descriptor_source)
)

print(
    "Ошибочных формул:",
    len(failed_formulas)
)

print(
    "Уникальных составов:",
    len(unique_compositions)
)

print(
    unique_compositions.head()
)

print("Формулы подготовлены", flush=True)


# In[43]:


try:
    elements_data = preset.elements_completed

    print(
        "База elements_completed найдена:",
        elements_data.shape,
        flush=True
    )

except RuntimeError as error:
    print(
        "База elements_completed отсутствует.",
        flush=True
    )

    print(
        "Выполните в отдельной ячейке:",
        "preset.sync('elements_completed')",
        flush=True
    )

    raise error


# In[45]:


from xenonpy.descriptor.compositions import (
    WeightedAverage,
    WeightedSum,
    WeightedVariance,
    MaxPooling,
    MinPooling,
)

print(
    "Отдельные калькуляторы импортированы",
    flush=True
)


# In[47]:


# ==================================================================
# Блок 10. Проверка отдельных калькуляторов
# ==================================================================

from xenonpy.descriptor.compositions import (
    WeightedAverage,
    WeightedSum,
    WeightedVariance,
    MaxPooling,
    MinPooling,
)


test_composition = (
    unique_compositions[
        "composition"
    ]
    .iloc[0]
)

test_formula = (
    unique_compositions[
        "Clean_formula"
    ]
    .iloc[0]
)

print(
    "Тестовая формула:",
    test_formula,
    flush=True
)

print(
    "Тестовый состав:",
    test_composition,
    flush=True
)


test_calculators = {
    "average": WeightedAverage(
        n_jobs=1,
        on_errors="raise",
        return_type="df"
    ),

    "sum": WeightedSum(
        n_jobs=1,
        on_errors="raise",
        return_type="df"
    ),

    "variance": WeightedVariance(
        n_jobs=1,
        on_errors="raise",
        return_type="df"
    ),

    "maximum": MaxPooling(
        n_jobs=1,
        on_errors="raise",
        return_type="df"
    ),

    "minimum": MinPooling(
        n_jobs=1,
        on_errors="raise",
        return_type="df"
    ),
}

test_results = {}

for calculator_name, calculator in test_calculators.items():
    print(
        f"\nНачало теста: {calculator_name}",
        flush=True
    )

    # threadpool_limits здесь намеренно не используется
    current_result = calculator.transform(
        [test_composition]
    )

    if not isinstance(
        current_result,
        pd.DataFrame
    ):
        current_result = pd.DataFrame(
            current_result,
            columns=calculator.feature_labels
        )

    current_result = current_result.reset_index(
        drop=True
    )

    test_results[
        calculator_name
    ] = current_result

    print(
        f"{calculator_name}: "
        f"{current_result.shape}",
        flush=True
    )

    print(
        f"Количество пропусков: "
        f"{current_result.isna().sum().sum()}",
        flush=True
    )

print(
    "\nВсе калькуляторы успешно прошли тест",
    flush=True
)


# In[49]:


# ==================================================================
# Блок 11. Пакетный расчёт дескрипторов
# ==================================================================

import gc


def calculate_in_batches(
    calculator,
    calculator_name,
    compositions,
    batch_size=10
):
    """
    Последовательно рассчитывает дескрипторы небольшими пакетами.

    После каждого пакета результат сохраняется в CSV.
    Это позволяет определить проблемный диапазон составов.
    """
    results = []

    number_of_compositions = len(
        compositions
    )

    print(
        f"\nКалькулятор: {calculator_name}",
        flush=True
    )

    print(
        f"Количество составов: "
        f"{number_of_compositions}",
        flush=True
    )

    for start in range(
        0,
        number_of_compositions,
        batch_size
    ):
        stop = min(
            start + batch_size,
            number_of_compositions
        )

        print(
            f"{calculator_name}: "
            f"начало пакета {start}:{stop}",
            flush=True
        )

        current_batch = (
            compositions
            .iloc[start:stop]
            .tolist()
        )

        # threadpool_limits не используем
        current_result = calculator.transform(
            current_batch
        )

        if not isinstance(
            current_result,
            pd.DataFrame
        ):
            current_result = pd.DataFrame(
                current_result,
                columns=calculator.feature_labels
            )

        current_result = (
            current_result
            .reset_index(drop=True)
        )

        checkpoint_file = (
            CHECKPOINT_DIRECTORY
            / (
                f"{calculator_name}_"
                f"{start:04d}_{stop:04d}.csv"
            )
        )

        current_result.to_csv(
            checkpoint_file,
            index=False,
            encoding="utf-8-sig"
        )

        results.append(
            current_result
        )

        print(
            f"{calculator_name}: "
            f"пакет {start}:{stop} сохранён",
            flush=True
        )

        del current_batch

        gc.collect()

    complete_result = pd.concat(
        results,
        ignore_index=True
    )

    print(
        f"{calculator_name}: расчёт завершён, "
        f"размер {complete_result.shape}",
        flush=True
    )

    return complete_result


# In[51]:


average_calculator = WeightedAverage(
    n_jobs=1,
    on_errors="nan",
    return_type="df"
)

average_descriptors = calculate_in_batches(
    calculator=average_calculator,
    calculator_name="average",
    compositions=unique_compositions["composition"],
    batch_size=10
)

print(
    "Средние дескрипторы рассчитаны:",
    average_descriptors.shape,
    flush=True
)


# In[53]:


sum_calculator = WeightedSum(
    n_jobs=1,
    on_errors="nan",
    return_type="df"
)

sum_descriptors = calculate_in_batches(
    calculator=sum_calculator,
    calculator_name="sum",
    compositions=unique_compositions["composition"],
    batch_size=10
)

print(
    "Суммарные дескрипторы рассчитаны:",
    sum_descriptors.shape,
    flush=True
)


# In[55]:


variance_calculator = WeightedVariance(
    n_jobs=1,
    on_errors="nan",
    return_type="df"
)

variance_descriptors = calculate_in_batches(
    calculator=variance_calculator,
    calculator_name="variance",
    compositions=unique_compositions["composition"],
    batch_size=10
)

print(
    "Дисперсионные дескрипторы рассчитаны:",
    variance_descriptors.shape,
    flush=True
)


# In[57]:


maximum_calculator = MaxPooling(
    n_jobs=1,
    on_errors="nan",
    return_type="df"
)

maximum_descriptors = calculate_in_batches(
    calculator=maximum_calculator,
    calculator_name="maximum",
    compositions=unique_compositions["composition"],
    batch_size=10
)

print(
    "Максимальные дескрипторы рассчитаны:",
    maximum_descriptors.shape,
    flush=True
)


# In[59]:


minimum_calculator = MinPooling(
    n_jobs=1,
    on_errors="nan",
    return_type="df"
)

minimum_descriptors = calculate_in_batches(
    calculator=minimum_calculator,
    calculator_name="minimum",
    compositions=unique_compositions["composition"],
    batch_size=10
)

print(
    "Минимальные дескрипторы рассчитаны:",
    minimum_descriptors.shape,
    flush=True
)


# In[61]:


xenon_descriptors_unique = pd.concat(
    [
        average_descriptors.reset_index(drop=True),
        sum_descriptors.reset_index(drop=True),
        variance_descriptors.reset_index(drop=True),
        maximum_descriptors.reset_index(drop=True),
        minimum_descriptors.reset_index(drop=True),
    ],
    axis=1
)

xenon_descriptors_unique = (
    xenon_descriptors_unique
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

print(
    "Размер таблицы до очистки:",
    xenon_descriptors_unique.shape
)

all_nan_columns = (
    xenon_descriptors_unique.columns[
        xenon_descriptors_unique
        .isna()
        .all()
    ]
    .tolist()
)

xenon_descriptors_unique = (
    xenon_descriptors_unique
    .drop(
        columns=all_nan_columns
    )
)

constant_columns = (
    xenon_descriptors_unique.columns[
        xenon_descriptors_unique
        .nunique(dropna=True) <= 1
    ]
    .tolist()
)

xenon_descriptors_unique = (
    xenon_descriptors_unique
    .drop(
        columns=constant_columns
    )
)

print(
    "Удалено пустых:",
    len(all_nan_columns)
)

print(
    "Удалено постоянных:",
    len(constant_columns)
)

print(
    "Итоговый размер:",
    xenon_descriptors_unique.shape,
    flush=True
)


# In[63]:


xenon_descriptors_unique = pd.concat(
    [
        average_descriptors.reset_index(drop=True),
        sum_descriptors.reset_index(drop=True),
        variance_descriptors.reset_index(drop=True),
        maximum_descriptors.reset_index(drop=True),
        minimum_descriptors.reset_index(drop=True),
    ],
    axis=1
)

xenon_descriptors_unique = (
    xenon_descriptors_unique
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

print(
    "Размер таблицы до очистки:",
    xenon_descriptors_unique.shape
)

all_nan_columns = (
    xenon_descriptors_unique.columns[
        xenon_descriptors_unique
        .isna()
        .all()
    ]
    .tolist()
)

xenon_descriptors_unique = (
    xenon_descriptors_unique
    .drop(
        columns=all_nan_columns
    )
)

constant_columns = (
    xenon_descriptors_unique.columns[
        xenon_descriptors_unique
        .nunique(dropna=True) <= 1
    ]
    .tolist()
)

xenon_descriptors_unique = (
    xenon_descriptors_unique
    .drop(
        columns=constant_columns
    )
)

print(
    "Удалено пустых:",
    len(all_nan_columns)
)

print(
    "Удалено постоянных:",
    len(constant_columns)
)

print(
    "Итоговый размер:",
    xenon_descriptors_unique.shape,
    flush=True
)


# In[65]:


unique_descriptor_table = pd.concat(
    [
        unique_compositions[
            ["Clean_formula"]
        ].reset_index(drop=True),

        xenon_descriptors_unique.reset_index(
            drop=True
        ),
    ],
    axis=1
)

metadata_columns = [
    "Original_index",
    "Compound",
    "Clean_formula",
    "Property",
    "Magnetic Field",
    "D_crystalite [nm]",
    "Reference",
    "delta_S_M",
    "log_delta_S_M",
    "Is_target_outlier",
]

metadata_columns = [
    column
    for column in metadata_columns
    if column in descriptor_source.columns
]

metadata = (
    descriptor_source[
        metadata_columns
    ]
    .reset_index(drop=True)
)

ml_dataset = metadata.merge(
    unique_descriptor_table,
    on="Clean_formula",
    how="left",
    validate="many_to_one"
)

descriptor_columns = (
    xenon_descriptors_unique
    .columns
    .tolist()
)

print(
    "Размер ML-таблицы:",
    ml_dataset.shape
)

print(
    "Экспериментальных строк:",
    len(ml_dataset)
)

print(
    "Уникальных составов:",
    ml_dataset["Clean_formula"].nunique()
)

print(
    "Количество дескрипторов:",
    len(descriptor_columns),
    flush=True
)


# In[67]:


removed_descriptors = pd.DataFrame({
    "Descriptor": (
        all_nan_columns
        + constant_columns
    ),
    "Reason": (
        ["all_NaN"] * len(all_nan_columns)
        + ["constant"] * len(constant_columns)
    )
})

with pd.ExcelWriter(
    OUTPUT_XLSX,
    engine="openpyxl"
) as writer:

    ml_dataset.to_excel(
        writer,
        sheet_name="ML_data",
        index=False
    )

    unique_descriptor_table.to_excel(
        writer,
        sheet_name="Unique_compositions",
        index=False
    )

    failed_formulas.to_excel(
        writer,
        sheet_name="Failed_formulas",
        index=False
    )

    removed_descriptors.to_excel(
        writer,
        sheet_name="Removed_descriptors",
        index=False
    )

ml_dataset.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print(
    "Excel сохранён:",
    OUTPUT_XLSX.resolve(),
    flush=True
)

print(
    "CSV сохранён:",
    OUTPUT_CSV.resolve(),
    flush=True
)


# In[ ]:




