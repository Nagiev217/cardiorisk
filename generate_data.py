"""
generate_data.py: запасной генератор учебного датасета data/heart.csv.

Основной способ получить данные: python download_data.py, он скачивает
настоящий датасет UCI. Этот скрипт нужен, если интернета нет. Он создаёт
синтетических пациентов с теми же колонками и с похожими на правду
зависимостями: возраст, давление, холестерин и прочее повышают риск.

Колонки (как в UCI Heart Disease):
    age      возраст, лет
    sex      пол (1 мужской, 0 женский)
    cp       тип боли в груди (0 типичная стенокардия, 1 атипичная,
             2 неангинозная боль, 3 бессимптомно)
    trestbps давление в покое, мм рт. ст.
    chol     холестерин, мг/дл
    fbs      сахар натощак больше 120 мг/дл (1 да, 0 нет)
    restecg  ЭКГ в покое (0 норма, 1 аномалия ST-T, 2 гипертрофия ЛЖ)
    thalach  максимальный пульс при нагрузке, уд/мин
    exang    стенокардия при нагрузке (1 да, 0 нет)
    oldpeak  депрессия сегмента ST при нагрузке, мм
    slope    наклон сегмента ST (0 восходящий, 1 плоский, 2 нисходящий)
    ca       сколько крупных сосудов видно при флюороскопии (0-3)
    thal     таллиевый тест (1 фиксированный дефект, 2 норма,
             3 обратимый дефект)
    target   ответ (1 есть заболевание, 0 нет)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Куда сохранить датасет.
DATA_PATH: Path = Path(__file__).parent / "data" / "heart.csv"

# Сколько пациентов (строк) сгенерировать.
N_SAMPLES: int = 1000

# С фиксированным зерном каждый запуск даёт один и тот же датасет.
RANDOM_SEED: int = 42


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Логистическая функция. Превращает любое число в вероятность от 0 до 1."""
    return 1.0 / (1.0 + np.exp(-z))


def generate(n: int, seed: int) -> pd.DataFrame:
    """Генерирует n пациентов и возвращает DataFrame."""
    rng = np.random.default_rng(seed)

    # --- Признаки пациента -------------------------------------------------
    # Возраст берём из нормального распределения со средним 54 и обрезаем до 29-77.
    age = np.clip(rng.normal(54, 9, n), 29, 77).round().astype(int)
    # Мужчин около 68%, примерно как в настоящем датасете.
    sex = rng.binomial(1, 0.68, n)
    # Тип боли в груди, у каждой категории своя вероятность.
    cp = rng.choice([0, 1, 2, 3], size=n, p=[0.17, 0.28, 0.08, 0.47])
    # С возрастом давление растёт, поэтому прибавляем слагаемое от age.
    trestbps = np.clip(rng.normal(131, 16, n) + 0.3 * (age - 54), 94, 200).round().astype(int)
    chol = np.clip(rng.normal(246, 50, n), 126, 564).round().astype(int)
    fbs = rng.binomial(1, 0.15, n)
    restecg = rng.choice([0, 1, 2], size=n, p=[0.48, 0.49, 0.03])
    # Максимальный пульс падает с возрастом (известная прикидка: 220 минус возраст).
    thalach = np.clip(205 - 0.9 * age + rng.normal(0, 18, n), 71, 202).round().astype(int)
    exang = rng.binomial(1, 0.33, n)
    oldpeak = np.clip(rng.exponential(1.0, n), 0, 6.2).round(1)
    slope = rng.choice([0, 1, 2], size=n, p=[0.47, 0.46, 0.07])
    ca = rng.choice([0, 1, 2, 3], size=n, p=[0.58, 0.21, 0.13, 0.08])
    thal = rng.choice([1, 2, 3], size=n, p=[0.06, 0.55, 0.39])

    # --- Скрытая формула риска ---------------------------------------------
    # Складываем факторы с весами (это логит). Положительный вес повышает
    # риск, отрицательный снижает.
    cp_effect = np.array([-0.7, -0.4, -0.3, 1.1])[cp]
    restecg_effect = np.array([0.0, 0.3, 0.5])[restecg]
    slope_effect = np.array([-0.5, 0.5, 0.6])[slope]
    thal_effect = np.array([0.0, 0.6, -0.7, 1.0])[thal]

    logit = (
        -2.0
        + 0.045 * (age - 54)
        + 0.6 * sex
        + cp_effect
        + 0.018 * (trestbps - 131)
        + 0.007 * (chol - 246)
        + 0.3 * fbs
        + restecg_effect
        - 0.03 * (thalach - 156)
        + 0.9 * exang
        + 0.6 * oldpeak
        + slope_effect
        + 0.75 * ca
        + thal_effect
    )
    # Логит переводим в вероятность, а дальше "бросаем монетку" с этой
    # вероятностью. Так в данных появляется шум: пациент с плохими
    # показателями иногда оказывается здоров, как и в жизни.
    target = rng.binomial(1, sigmoid(logit))

    return pd.DataFrame(
        {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
            "thal": thal, "target": target,
        }
    )


def main() -> None:
    """Генерируем датасет и пишем его в CSV."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate(N_SAMPLES, RANDOM_SEED)
    df.to_csv(DATA_PATH, index=False)
    print(f"Датасет сохранён: {DATA_PATH} ({len(df)} строк)")
    print(f"Доля пациентов с заболеванием: {df['target'].mean():.1%}")


if __name__ == "__main__":
    main()
