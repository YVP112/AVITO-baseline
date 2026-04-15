# Avito Services Splitter

ML-сервис для задачи RNC: найти в объявлении самостоятельные услуги, решить `shouldSplit` и подготовить черновики дополнительных объявлений.

Финальное решение в проекте: `TF-IDF word/char features + OneVsRest Logistic Regression + бинарный shouldSplit classifier + аккуратная post-processing логика для объявлений про ремонт под ключ`.

## Формат Ответа

В итоговом файле `response` лежит JSON:

```json
{
  "itemId": 5002,
  "detectedMcIds": [102, 103, 106],
  "shouldSplit": true,
  "drafts": [
    {
      "mcId": 102,
      "mcTitle": "Сантехника",
      "text": "Выполняем сантехника отдельно. Уточняйте детали и стоимость по объекту."
    }
  ]
}
```

## Что Внутри

```text
app/                  FastAPI API
data/raw/             обучающий датасет и справочник микрокатегорий
outputs/              заполненный тестовый файл и локальные артефакты
rnc_test_responses.csv готовый ответ на тестовые запросы организаторов
src/final_model.py    выбранная итоговая модель
src/ml_baseline.py    обучение TF-IDF + Logistic Regression
src/evaluate.py       локальные метрики
run_baseline.py       train -> validate -> predict test -> metrics
run_make_submission.py генерация response для rnc_test.csv
docker-compose.yml    запуск API и evaluation
Dockerfile            образ проекта
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Если виртуальное окружение уже создано:

```bash
source .venv/bin/activate
```

## Локальный Запуск

Посчитать локальные метрики и сохранить предсказания:

```bash
python run_baseline.py
```

Скрипт обучает итоговую модель на `train`, предсказывает `val/test` и сохраняет:

```text
outputs/val_predictions.csv
outputs/test_predictions.csv
outputs/val_predictions.jsonl
outputs/test_predictions.jsonl
outputs/error_analysis.csv
```

Собрать файл ответов для тестового файла организаторов:

```bash
python run_make_submission.py /path/to/rnc_test.csv
```

Результат пересборки:

```text
outputs/rnc_test_responses.csv
outputs/rnc_test_audit.csv
```

Готовый статичный ответ на тестовые запросы лежит в корне репозитория:

```text
rnc_test_responses.csv
```

Путь к тесту также можно передать через переменную окружения:

```bash
RNC_TEST_PATH=/path/to/rnc_test.csv python run_make_submission.py
```

## Docker

Единый запуск API:

```bash
docker compose up --build
```

API будет доступен на:

```text
http://localhost:8000
```

Локальная оценка через compose:

```bash
docker compose run --rm eval
```

## API

Запуск без Docker:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://localhost:8000/health
```

Swagger UI:

```text
http://localhost:8000/docs
```

Пример запроса:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": 1001,
    "source_mc_id": 101,
    "description": "Мастер на час. Отдельно выполняю сантехнику, электрику и поклейку обоев."
  }'
```

## Метрики

Основная локальная команда:

```bash
python run_baseline.py
```

Проект считает:

| Метрика | Что показывает |
| --- | --- |
| `precision_micro` | доля корректных предложенных draft-категорий |
| `recall_micro` | доля найденных целевых draft-категорий |
| `f1_micro` | главный баланс precision/recall по микрокатегориям |
| `should_split_f1` | качество бинарного решения split / no split |
| `coverage_recall` | покрытие целевых микрокатегорий |
| `tn/fp/fn/tp` | разбор бинарной split-задачи |

Текущий локальный ориентир:

```text
validation f1_micro: 0.9135
validation should_split_f1: 0.9784
```

Актуальные цифры печатает `python run_baseline.py`.

## Как Работает Модель

1. Текст объявления очищается от SEO-хвостов.
2. Модель строит word-level и char-level TF-IDF признаки.
3. Multi-label классификатор предсказывает `detectedMcIds`.
4. Отдельный классификатор предсказывает вероятность `shouldSplit`.
5. Индивидуальные пороги подбираются на holdout-части train.
6. Post-processing подавляет услуги, которые явно описаны как часть ремонта под ключ.
7. Post-processing усиливает объявления-списки, где автор явно продаёт несколько услуг.
8. Для выбранных микрокатегорий генерируются тексты черновиков.

## GitLab CI

`.gitlab-ci.yml` запускает:

```bash
python run_baseline.py
```

и сохраняет validation/test outputs как артефакты.
