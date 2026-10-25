"""
train.py: обучение моделей для прогноза риска сердечно-сосудистых заболеваний.

Порядок работы:
    1. Читаем data/heart.csv.
    2. Собираем Pipeline: числовые признаки масштабируем StandardScaler'ом,
       категориальные кодируем OneHotEncoder'ом, дальше идёт классификатор.
    3. Обучаем LogisticRegression, RandomForest и GradientBoosting
       и печатаем таблицу метрик (Accuracy, Precision, Recall, F1).
    4. Для модели с лучшим F1 на кросс-валидации подбираем гиперпараметры
       через GridSearchCV.
    5. Считаем важность признаков и сохраняем её в features.json вместе
       со статистикой по датасету (она нужна сайту для объяснений).
    6. Сохраняем готовый Pipeline в model.pkl.

Запуск:
    python train.py
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Константы и пути к файлам
# ---------------------------------------------------------------------------

# Пути строим от папки, где лежит сам скрипт. Так он найдёт файлы,
# даже если запустить его из другой директории.
BASE_DIR: Path = Path(__file__).parent
DATA_PATH: Path = BASE_DIR / "data" / "heart.csv"
MODEL_PATH: Path = BASE_DIR / "model.pkl"
FEATURES_PATH: Path = BASE_DIR / "features.json"

# Колонка с ответом: 1 значит, что заболевание есть, 0 значит, что нет.
TARGET: str = "target"

# Непрерывные признаки. StandardScaler приводит их к среднему 0
# и стандартному отклонению 1, иначе холестерин (сотни) "перевешивал" бы
# oldpeak (единицы) просто из-за масштаба.
NUMERIC_FEATURES: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Здесь числа обозначают категории, а не величины: cp=3 не в три раза
# "больше", чем cp=1. Поэтому их кодируем через OneHotEncoder, и каждая
# категория превращается в отдельный столбец из нулей и единиц.
CATEGORICAL_FEATURES: list[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Зерно случайности фиксируем, чтобы при повторном запуске получались
# те же самые цифры.
RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Подготовка данных
# ---------------------------------------------------------------------------

def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Читает CSV и делит его на признаки X и ответ y."""
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл {path}. Положите heart.csv в папку data/ "
            f"или выполните: python download_data.py"
        )
    df = pd.read_csv(path)
    # Строки с пропусками выкидываем, на NaN модель обучаться не умеет.
    df = df.dropna()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer применяет к разным колонкам разные преобразования
    и потом склеивает всё обратно в одну матрицу.
    """
    return ColumnTransformer(
        transformers=[
            # Формат: (имя шага, преобразователь, список колонок).
            ("num", StandardScaler(), NUMERIC_FEATURES),
            # handle_unknown="ignore": если придёт категория, которой не было
            # в обучающих данных, она закодируется нулями без ошибки.
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model: ClassifierMixin) -> Pipeline:
    """
    Склеивает предобработку и модель в один объект.

    Зачем так: на кросс-валидации scaler каждый раз обучается только на
    своей обучающей части, и данные из проверочной части в него не
    попадают (нет утечки). А во Flask потом достаточно вызвать
    predict_proba, вручную готовить данные не нужно.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


# ---------------------------------------------------------------------------
# Модели и сетки гиперпараметров
# ---------------------------------------------------------------------------

def get_models() -> dict[str, ClassifierMixin]:
    """Три модели для сравнения, пока с настройками по умолчанию."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


# Варианты гиперпараметров для GridSearchCV. Приставка "model__" говорит,
# что параметр относится к шагу Pipeline под названием "model".
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "LogisticRegression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],          # регуляризация: чем меньше C, тем она сильнее
        "model__class_weight": [None, "balanced"],   # компенсировать ли разницу в размерах классов
    },
    "RandomForest": {
        "model__n_estimators": [100, 300],           # сколько деревьев в лесу
        "model__max_depth": [None, 5, 10],           # насколько глубоким может быть дерево
        "model__min_samples_leaf": [1, 3, 5],        # сколько объектов минимум в листе
    },
    "GradientBoosting": {
        "model__n_estimators": [100, 200],           # сколько деревьев строит бустинг
        "model__learning_rate": [0.05, 0.1],         # шаг обучения
        "model__max_depth": [2, 3],                  # глубина одного дерева
    },
}


# ---------------------------------------------------------------------------
# Оценка качества
# ---------------------------------------------------------------------------

def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """
    Метрики на тестовой выборке.

    Accuracy:  какая доля ответов правильная.
    Precision: из тех, кого модель назвала больными, сколько больны на самом деле.
    Recall:    из всех больных сколько модель нашла. Для медицины это важнее
               всего, потому что пропущенный больной хуже ложной тревоги.
    F1:        среднее гармоническое Precision и Recall.
    ROC-AUC:   насколько хорошо модель ранжирует пациентов по вероятности
               (1.0 идеально, 0.5 как подбрасывание монетки).
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # второй столбец = вероятность класса 1
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> pd.DataFrame:
    """
    Обучает все три модели и собирает их метрики в таблицу.

    Кроме тестовых метрик считаем F1 на кросс-валидации (CV F1). Лучшую
    модель выбираем по нему. Если выбирать по тесту, тест перестанет быть
    независимой проверкой.
    """
    rows: list[dict[str, Any]] = []
    for name, model in get_models().items():
        pipeline = build_pipeline(model)
        # Кросс-валидация на 5 частях обучающей выборки.
        cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1").mean()
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        rows.append({"Model": name, "CV F1": cv_f1, **metrics})
    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------------
# Важность признаков и статистика для объяснений на сайте
# ---------------------------------------------------------------------------

def extract_importances(pipeline: Pipeline) -> dict[str, float]:
    """
    Достаёт важность признаков из модели и пересчитывает её на исходные
    колонки датасета.

    После OneHotEncoder один признак cp превращается в четыре столбца
    (cp_0 ... cp_3), и модель знает важность каждого из них по отдельности.
    Чтобы получить одну цифру на cp, складываем важности его столбцов.
    """
    model = pipeline.named_steps["model"]
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]

    # У деревьев (RandomForest, GradientBoosting) есть готовый атрибут
    # feature_importances_. У логистической регрессии его нет, зато есть
    # коэффициенты coef_. Признаки у нас стандартизированы, поэтому модуль
    # коэффициента можно считать силой влияния.
    if hasattr(model, "feature_importances_"):
        raw: np.ndarray = model.feature_importances_
    else:
        raw = np.abs(model.coef_[0])

    # Сколько one-hot столбцов получилось у каждого категориального признака.
    encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]
    cat_sizes: list[int] = [len(categories) for categories in encoder.categories_]

    importances: dict[str, float] = {}
    position = 0
    # Столбцы идут в том же порядке, что и шаги ColumnTransformer:
    # сначала числовые (по одному столбцу на признак), потом one-hot.
    for feature in NUMERIC_FEATURES:
        importances[feature] = float(raw[position])
        position += 1
    for feature, size in zip(CATEGORICAL_FEATURES, cat_sizes):
        importances[feature] = float(raw[position:position + size].sum())
        position += size

    # Делим на сумму, чтобы всё вместе давало 1 (потом удобно показывать в %).
    total = sum(importances.values())
    importances = {name: value / total for name, value in importances.items()}
    # Самые важные признаки ставим первыми.
    return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))


def compute_feature_stats(X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, Any]]:
    """
    Статистика по признакам, по которой сайт объясняет прогноз пациенту.

    Для числовых признаков сохраняем среднее, стандартное отклонение и
    направление: +1, если чем больше значение, тем выше риск, и -1, если
    наоборот. Для категориальных сохраняем долю больных в каждой категории.
    """
    stats: dict[str, dict[str, Any]] = {}
    for feature in NUMERIC_FEATURES:
        correlation = X[feature].corr(y)  # корреляция Пирсона с target
        stats[feature] = {
            "type": "numeric",
            "mean": float(X[feature].mean()),
            "std": float(X[feature].std()),
            "direction": 1 if correlation >= 0 else -1,
        }
    for feature in CATEGORICAL_FEATURES:
        # Среднее от target внутри группы и есть доля больных в этой категории.
        rates = y.groupby(X[feature]).mean()
        stats[feature] = {
            "type": "categorical",
            # В JSON ключи бывают только строками, поэтому str(category).
            "risk_rate": {str(category): float(rate) for category, rate in rates.items()},
        }
    return stats


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main() -> None:
    """Весь процесс по шагам: данные, сравнение, подбор параметров, сохранение."""
    print("=" * 70)
    print("1. Загрузка данных")
    X, y = load_data(DATA_PATH)
    print(f"   Пациентов: {len(X)}, признаков: {X.shape[1]}, доля больных: {y.mean():.1%}")

    # 80% данных на обучение, 20% на тест. stratify=y нужен, чтобы доля
    # больных в обеих частях была одинаковой.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("=" * 70)
    print("2. Сравнение трёх моделей (метрики на тестовой выборке)")
    comparison = compare_models(X_train, y_train, X_test, y_test, cv)
    print(comparison.round(3).to_string())

    # Берём модель с самым высоким F1 на кросс-валидации.
    best_name: str = str(comparison["CV F1"].idxmax())
    print(f"\n   Лучшая модель по CV F1: {best_name}")

    print("=" * 70)
    print(f"3. Подбор гиперпараметров {best_name} через GridSearchCV")
    search = GridSearchCV(
        estimator=build_pipeline(get_models()[best_name]),
        param_grid=PARAM_GRIDS[best_name],
        scoring="f1",
        cv=cv,
        n_jobs=-1,  # считать на всех ядрах процессора
    )
    search.fit(X_train, y_train)
    best_pipeline: Pipeline = search.best_estimator_
    print(f"   Лучшие параметры: {search.best_params_}")
    print(f"   CV F1 до подбора: {comparison.loc[best_name, 'CV F1']:.3f}, "
          f"после: {search.best_score_:.3f}")

    final_metrics = evaluate(best_pipeline, X_test, y_test)
    print("\n   Метрики итоговой модели на тесте:")
    for metric, value in final_metrics.items():
        print(f"   {metric:<10} {value:.3f}")
    print("\n" + classification_report(y_test, best_pipeline.predict(X_test),
                                       target_names=["Здоров", "Болен"]))

    print("=" * 70)
    print("4. Важность признаков (Feature Importances)")
    importances = extract_importances(best_pipeline)
    for feature, weight in importances.items():
        bar = "#" * int(weight * 100)
        print(f"   {feature:<10} {weight:6.1%} {bar}")

    # Всё, что понадобится сайту, складываем в features.json.
    features_info: dict[str, Any] = {
        "model_name": best_name,
        "best_params": search.best_params_,
        "metrics": final_metrics,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "importances": importances,
        "base_rate": float(y.mean()),  # какая доля пациентов в датасете больна
        "stats": compute_feature_stats(X, y),
    }
    FEATURES_PATH.write_text(json.dumps(features_info, ensure_ascii=False, indent=2), encoding="utf-8")

    # Pipeline сохраняем целиком, вместе с предобработкой, одним файлом.
    joblib.dump(best_pipeline, MODEL_PATH)

    print("=" * 70)
    print(f"Модель сохранена:  {MODEL_PATH}")
    print(f"Признаки сохранены: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
