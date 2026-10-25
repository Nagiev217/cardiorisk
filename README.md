# CardioRisk — прогноз риска сердечно-сосудистых заболеваний

Веб-приложение на Flask + scikit-learn (классический ML, работает на CPU).

## Данные

Используется настоящий датасет **UCI Heart Disease (Cleveland)**: 297 пациентов
(303 минус 6 строк с пропусками), 13 признаков, 46% с заболеванием.
Источник: https://archive.ics.uci.edu/dataset/45/heart+disease
Скрипт `download_data.py` скачивает его и приводит к формату `data/heart.csv`.

## Структура

```
data/heart.csv        датасет UCI Heart Disease (Cleveland)
download_data.py      скачивание и конвертация датасета UCI
generate_data.py      запасной вариант: синтетический датасет того же формата (если нет интернета)
train.py              обучение и сравнение 3 моделей, GridSearchCV -> model.pkl, features.json
app.py                Flask-приложение + SQLite (database.db создаётся автоматически)
templates/            base.html, index.html, result.html, history.html
requirements.txt
```

## Запуск (Windows)

```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_data.py      # только если data/heart.csv ещё нет
python train.py
python app.py
```

Открыть http://127.0.0.1:5000

## Результаты (тестовая выборка, 60 пациентов)

LogisticRegression (C=0.1, class_weight=balanced): Accuracy 0.85, Precision 0.88,
Recall 0.79, F1 0.83, ROC-AUC 0.94.

## REST API

```
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" ^
  -d "{\"age\":63,\"sex\":1,\"cp\":3,\"trestbps\":145,\"chol\":290,\"fbs\":1,\"restecg\":1,\"thalach\":125,\"exang\":1,\"oldpeak\":2.3,\"slope\":1,\"ca\":2,\"thal\":3}"
```
