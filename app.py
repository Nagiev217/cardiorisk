"""
app.py: Flask-приложение для прогноза риска сердечно-сосудистых заболеваний.

Маршруты:
    GET  /             форма для данных пациента
    POST /predict      прогноз по форме, запись в БД, страница с результатом
    GET  /history      таблица всех прошлых запросов
    POST /api/predict  REST API: принимает JSON и возвращает JSON

Сначала нужно обучить модель:  python train.py
Потом запустить сайт:          python app.py
"""

import json
import math
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Пути к файлам
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).parent
MODEL_PATH: Path = BASE_DIR / "model.pkl"
FEATURES_PATH: Path = BASE_DIR / "features.json"
DB_PATH: Path = BASE_DIR / "database.db"

# ---------------------------------------------------------------------------
# Поля формы: подпись, тип, допустимый диапазон или список вариантов.
# Всё описано в одном месте. Отсюда шаблон index.html строит форму,
# validate_input берёт правила проверки, а explain берёт подписи.
# Если поменять диапазон здесь, он поменяется везде.
# ---------------------------------------------------------------------------

FIELDS: dict[str, dict[str, Any]] = {
    "age": {"label": "Возраст", "unit": "лет", "type": "int", "min": 1, "max": 120, "step": 1,
            "hint": "Полных лет"},
    "sex": {"label": "Пол", "type": "choice", "options": {1: "Мужской", 0: "Женский"}},
    "cp": {"label": "Тип боли в груди", "type": "choice",
           "options": {0: "Типичная стенокардия", 1: "Атипичная стенокардия",
                       2: "Неангинозная боль", 3: "Бессимптомно"}},
    "trestbps": {"label": "Давление в покое", "unit": "мм рт. ст.", "type": "int",
                 "min": 50, "max": 250, "step": 1, "hint": "Систолическое (верхнее)"},
    "chol": {"label": "Холестерин", "unit": "мг/дл", "type": "int",
             "min": 100, "max": 700, "step": 1, "hint": "Общий холестерин в сыворотке"},
    "fbs": {"label": "Сахар натощак > 120 мг/дл", "type": "choice", "options": {0: "Нет", 1: "Да"}},
    "restecg": {"label": "ЭКГ в покое", "type": "choice",
                "options": {0: "Норма", 1: "Аномалия ST-T", 2: "Гипертрофия левого желудочка"}},
    "thalach": {"label": "Максимальный пульс", "unit": "уд/мин", "type": "int",
                "min": 50, "max": 250, "step": 1, "hint": "Достигнутый при нагрузочной пробе"},
    "exang": {"label": "Стенокардия при нагрузке", "type": "choice", "options": {0: "Нет", 1: "Да"}},
    "oldpeak": {"label": "Депрессия ST", "unit": "мм", "type": "float",
                "min": 0, "max": 10, "step": 0.1, "hint": "Снижение сегмента ST при нагрузке"},
    "slope": {"label": "Наклон сегмента ST", "type": "choice",
              "options": {0: "Восходящий", 1: "Плоский", 2: "Нисходящий"}},
    "ca": {"label": "Поражённые сосуды (флюороскопия)", "type": "choice",
           "options": {0: "0", 1: "1", 2: "2", 3: "3"}},
    "thal": {"label": "Таллиевый тест", "type": "choice",
             "options": {1: "Фиксированный дефект", 2: "Норма", 3: "Обратимый дефект"}},
}

# Уровни риска: (верхняя граница вероятности, код для цвета, подпись).
# До 30% низкий, от 30 до 60% умеренный, выше 60% высокий.
RISK_LEVELS: list[tuple[float, str, str]] = [
    (0.30, "low", "Низкий риск"),
    (0.60, "medium", "Умеренный риск"),
    (1.01, "high", "Высокий риск"),
]

# ---------------------------------------------------------------------------
# Загрузка модели. Происходит один раз при старте сервера, а не на каждый
# запрос, иначе каждый прогноз заново читал бы файл с диска.
# ---------------------------------------------------------------------------


def load_artifacts() -> tuple[Pipeline, dict[str, Any]]:
    """Загружает Pipeline из model.pkl и метаданные из features.json (их создаёт train.py)."""
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError("Модель не найдена. Сначала выполните: python train.py")
    model: Pipeline = joblib.load(MODEL_PATH)
    features: dict[str, Any] = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    return model, features


MODEL, FEATURES_INFO = load_artifacts()

app = Flask(__name__)
# Без этой настройки jsonify отдаёт русские буквы в виде \uXXXX.
app.json.ensure_ascii = False

# ---------------------------------------------------------------------------
# База данных SQLite
# ---------------------------------------------------------------------------


def get_connection() -> sqlite3.Connection:
    """Открывает соединение с БД. Благодаря row_factory к полям строки можно обращаться по имени."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    """Создаёт таблицу predictions, если её ещё нет."""
    # closing() закрывает соединение в конце блока. "with connection"
    # сам делает COMMIT, а если внутри случилась ошибка, то ROLLBACK.
    with closing(get_connection()) as connection, connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT    NOT NULL,
                inputs    TEXT    NOT NULL,  -- введённые данные в виде JSON
                risk_prob REAL    NOT NULL,  -- вероятность заболевания от 0 до 1
                result    TEXT    NOT NULL   -- текстовый итог, например "Высокий риск"
            )
            """
        )


def save_prediction(inputs: dict[str, float | int], risk_prob: float, result: str) -> int:
    """Записывает прогноз в БД и возвращает id новой строки."""
    with closing(get_connection()) as connection, connection:
        # Значения подставляются через "?", а не склейкой строк.
        # Так SQL-инъекция не пройдёт.
        cursor = connection.execute(
            "INSERT INTO predictions (timestamp, inputs, risk_prob, result) VALUES (?, ?, ?, ?)",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                json.dumps(inputs, ensure_ascii=False),
                risk_prob,
                result,
            ),
        )
        return int(cursor.lastrowid)


def fetch_history() -> list[dict[str, Any]]:
    """Возвращает все прогнозы, самые новые сверху."""
    with closing(get_connection()) as connection:
        rows = connection.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    history: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record["inputs"] = json.loads(record["inputs"])  # строку JSON обратно в словарь
        history.append(record)
    return history


# ---------------------------------------------------------------------------
# Проверка входных данных
# ---------------------------------------------------------------------------


def validate_input(raw: dict[str, Any]) -> tuple[dict[str, float | int], dict[str, str]]:
    """
    Проверяет данные пациента, которые пришли из формы или через API.

    Возвращает два словаря: values с уже приведёнными к нужному типу
    значениями и errors вида {поле: текст ошибки}. Если ошибок нет,
    errors пустой.

    В HTML у полей тоже стоят min/max, но полагаться на них нельзя:
    их легко обойти через инструменты разработчика, а API форму вообще
    не использует. Поэтому сервер проверяет всё сам.
    """
    values: dict[str, float | int] = {}
    errors: dict[str, str] = {}

    for name, spec in FIELDS.items():
        value = raw.get(name)
        if value is None or str(value).strip() == "":
            errors[name] = "Поле обязательно для заполнения"
            continue
        try:
            # Из формы любое значение приходит строкой, так что переводим в число сами.
            # Запятую меняем на точку, чтобы "1,5" тоже принималось.
            number = float(str(value).replace(",", "."))
        except ValueError:
            errors[name] = "Введите число"
            continue
        # float() пропускает "inf" и "nan", такие значения отсекаем отдельно.
        if not math.isfinite(number):
            errors[name] = "Введите конечное число"
            continue

        if spec["type"] == "choice":
            if not number.is_integer() or int(number) not in spec["options"]:
                errors[name] = "Выберите значение из списка"
                continue
            values[name] = int(number)
        else:
            if spec["type"] == "int" and not number.is_integer():
                errors[name] = "Введите целое число"
                continue
            if not spec["min"] <= number <= spec["max"]:
                errors[name] = f"Допустимый диапазон: {spec['min']}–{spec['max']}"
                continue
            values[name] = int(number) if spec["type"] == "int" else round(number, 1)

    return values, errors


# ---------------------------------------------------------------------------
# Прогноз и объяснение
# ---------------------------------------------------------------------------


def get_risk_level(probability: float) -> tuple[str, str]:
    """По вероятности подбирает уровень риска: код для цвета и подпись."""
    for threshold, code, label in RISK_LEVELS:
        if probability < threshold:
            return code, label
    return RISK_LEVELS[-1][1], RISK_LEVELS[-1][2]


def format_value(name: str, value: float | int) -> str:
    """Делает значение понятным человеку: cp=3 станет "Бессимптомно", а 250 станет "250 мг/дл"."""
    spec = FIELDS[name]
    if spec["type"] == "choice":
        return spec["options"][int(value)]
    return f"{value} {spec.get('unit', '')}".strip()


def explain(values: dict[str, float | int], top_n: int = 3) -> list[dict[str, Any]]:
    """
    Ищет top_n признаков, которые сильнее всего повлияли на прогноз
    для конкретного пациента.

    Вклад признака считаем так: общая важность признака в модели
    (из features.json), умноженная на то, насколько значение пациента
    отличается от среднего по датасету.

    Для числового признака отличие равно z-оценке (x - среднее) / σ,
    умноженной на направление (+1 или -1). Например, холестерин на 2σ выше
    среднего при направлении +1 даёт +2, то есть риск повышается.

    Для категориального признака берём долю больных в категории пациента,
    вычитаем общую долю больных и делим на σ бинарной величины. Например,
    если среди пациентов с cp="Бессимптомно" больных 73%, а в среднем 46%,
    отличие получится положительным.

    Плюс означает, что признак повышает риск, минус означает, что снижает.
    """
    importances: dict[str, float] = FEATURES_INFO["importances"]
    stats: dict[str, dict[str, Any]] = FEATURES_INFO["stats"]
    base_rate: float = FEATURES_INFO["base_rate"]
    # Стандартное отклонение величины, которая равна 1 с вероятностью base_rate.
    # "or 1.0" страхует от деления на ноль.
    base_std = math.sqrt(base_rate * (1 - base_rate)) or 1.0

    contributions: list[dict[str, Any]] = []
    for name, value in values.items():
        feature_stats = stats[name]
        if feature_stats["type"] == "numeric":
            std = feature_stats["std"] or 1.0
            deviation = (value - feature_stats["mean"]) / std * feature_stats["direction"]
        else:
            rate = feature_stats["risk_rate"].get(str(value), base_rate)
            deviation = (rate - base_rate) / base_std
        contributions.append({
            "feature": name,
            "label": FIELDS[name]["label"],
            "value": format_value(name, value),
            "score": importances.get(name, 0.0) * deviation,
        })

    # Сортируем по модулю. Сильное снижение риска нам тоже интересно, не только повышение.
    contributions.sort(key=lambda item: abs(item["score"]), reverse=True)
    top = contributions[:top_n]

    # Какую часть от суммы топ-факторов составляет каждый. Это ширина полоски на странице.
    total = sum(abs(item["score"]) for item in top) or 1.0
    for item in top:
        item["increases_risk"] = item["score"] > 0
        item["share"] = round(abs(item["score"]) / total * 100)
        item["score"] = round(item["score"], 4)
    return top


def make_prediction(values: dict[str, float | int]) -> dict[str, Any]:
    """Считает прогноз, пишет его в БД и возвращает всё, что нужно для вывода."""
    # Pipeline ждёт DataFrame с теми же колонками и в том же порядке, что при обучении.
    columns = FEATURES_INFO["numeric_features"] + FEATURES_INFO["categorical_features"]
    frame = pd.DataFrame([values])[columns]
    # predict_proba отдаёт [[P(здоров), P(болен)]], нам нужно второе число.
    probability = float(MODEL.predict_proba(frame)[0, 1])
    level, result = get_risk_level(probability)
    record_id = save_prediction(values, probability, result)
    return {
        "id": record_id,
        "risk_prob": round(probability, 4),
        "risk_percent": round(probability * 100, 1),
        "level": level,
        "result": result,
        "top_factors": explain(values),
    }


# ---------------------------------------------------------------------------
# Маршруты
# ---------------------------------------------------------------------------


@app.route("/", methods=["GET"])
def index() -> str:
    """Главная страница с формой."""
    return render_template("index.html", fields=FIELDS, values={}, errors={})


@app.route("/predict", methods=["POST"])
def predict() -> str | tuple[str, int]:
    """Обрабатывает отправленную форму."""
    values, errors = validate_input(request.form.to_dict())
    if errors:
        # Показываем форму ещё раз, с ошибками и с тем, что пользователь уже
        # ввёл. Заполнять всё с нуля из-за одной опечатки никто не хочет.
        return render_template("index.html", fields=FIELDS, values=request.form, errors=errors), 400
    prediction = make_prediction(values)
    inputs_view = {FIELDS[name]["label"]: format_value(name, value) for name, value in values.items()}
    return render_template("result.html", prediction=prediction, inputs=inputs_view,
                           model_name=FEATURES_INFO["model_name"])


@app.route("/history", methods=["GET"])
def history() -> str:
    """Таблица со всеми прошлыми запросами."""
    return render_template("history.html", history=fetch_history(), fields=FIELDS,
                           format_value=format_value)


@app.route("/api/predict", methods=["POST"])
def api_predict() -> tuple[Any, int]:
    """
    REST API, чтобы прогноз можно было получать из других программ.

    Пример запроса:
        POST /api/predict
        Content-Type: application/json
        {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, ...}

    Ответ 200: {"risk_prob": 0.83, "risk_percent": 83.0, "level": "high", ...}
    Ответ 400: {"errors": {"age": "Допустимый диапазон: 1–120"}}
    """
    # С silent=True битый JSON не роняет запрос исключением, а просто даёт None.
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"errors": {"body": "Ожидается JSON-объект"}}), 400
    values, errors = validate_input(payload)
    if errors:
        return jsonify({"errors": errors}), 400
    return jsonify(make_prediction(values)), 200


# Таблицу создаём прямо при импорте модуля. Тогда она точно будет на месте,
# как бы ни запустили приложение: через "python app.py" или через "flask run".
init_db()

if __name__ == "__main__":
    # debug=True перезапускает сервер при изменении кода и показывает
    # подробные ошибки. На настоящем сервере его надо выключать.
    app.run(debug=True)
