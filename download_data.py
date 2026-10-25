"""
download_data.py: скачивает датасет UCI Heart Disease (Cleveland) и
сохраняет его в data/heart.csv в том виде, который ждёт train.py.

Откуда данные: UCI Machine Learning Repository, Heart Disease Data Set,
    https://archive.ics.uci.edu/dataset/45/heart+disease
Авторы: Andras Janosi, William Steinbrunn, Matthias Pfisterer,
Robert Detrano (Cleveland Clinic Foundation, 1988).

В исходном файле некоторые признаки закодированы по-другому, поэтому
при конвертации их пересчитываем:
    cp     1..4 переводим в 0..3   (0 типичная стенокардия, ..., 3 бессимптомно)
    slope  1..3 переводим в 0..2   (0 восходящий, 1 плоский, 2 нисходящий)
    thal   3/6/7 переводим в 2/1/3 (2 норма, 1 фиксированный дефект, 3 обратимый)
    num    0..4 превращаем в target 0/1 (0 здоров, 1..4 есть заболевание)
Пропуски в файле обозначены знаком "?". Таких строк 6, их удаляем.
"""

import io
import urllib.request
from pathlib import Path

import pandas as pd

URL: str = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
DATA_PATH: Path = Path(__file__).parent / "data" / "heart.csv"

# Заголовка в файле нет, так что названия колонок задаём сами.
# Порядок взят из описания датасета (heart-disease.names).
COLUMNS: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def download(url: str) -> str:
    """Скачивает файл по ссылке и возвращает его текст."""
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def convert(raw_text: str) -> pd.DataFrame:
    """Переводит исходный текст UCI в DataFrame с нашей кодировкой признаков."""
    # na_values="?" говорит pandas, что вопросительный знак значит пропуск.
    df = pd.read_csv(io.StringIO(raw_text), header=None, names=COLUMNS, na_values="?")
    df = df.dropna()

    df["cp"] = df["cp"] - 1
    df["slope"] = df["slope"] - 1
    df["thal"] = df["thal"].map({3: 2, 6: 1, 7: 3})
    # Степени заболевания от 1 до 4 объединяем в один класс "болен".
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    # Из-за пропусков pandas прочитал часть колонок как float.
    # Возвращаем им целый тип. Дробным остаётся только oldpeak.
    int_columns = [column for column in df.columns if column != "oldpeak"]
    df[int_columns] = df[int_columns].astype(int)
    return df


def main() -> None:
    """Скачиваем, конвертируем, сохраняем."""
    print(f"Скачиваю {URL}")
    df = convert(download(URL))
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Датасет сохранён: {DATA_PATH} ({len(df)} строк)")
    print(f"Доля пациентов с заболеванием: {df['target'].mean():.1%}")


if __name__ == "__main__":
    main()
