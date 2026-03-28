# Avito Services Splitter

Решение для задачи автоматического выделения самостоятельных услуг и генерации черновиков объявлений в категории `Ремонт и отделка`.

Проект содержит два независимых подхода:
- `heuristic baseline` — основной production-like подход на словаре и правилах
- `ml baseline` — сравнительный ML-подход на `TF-IDF + Logistic Regression`

## Что делает решение

На вход подается объявление с исходной микрокатегорией и текстом описания.

Система должна:
- определить, какие микрокатегории услуг упомянуты в тексте
- понять, нужно ли создавать дополнительные черновики объявлений
- если нужно, сформировать список черновиков

Формат ответа соответствует ТЗ:

```json
{
  "detectedMcIds": [101, 102],
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

## Структура проекта

```text
hack/
  data/raw/
    rnc_dataset.csv
    rnc_dataset.jsonl
    rnc_dataset_README.txt
    rnc_mic_key_phrases.csv
  outputs/
    val_predictions.csv
    test_predictions.csv
    val_predictions.jsonl
    test_predictions.jsonl
    baseline_comparison.csv
  src/
    config.py
    data_loader.py
    text_preprocessing.py
    dictionary_matcher.py
    split_scorrer.py
    draft_generator.py
    predictor.py
    ml_baseline.py
    evaluate.py
  dashboard.py
  run_baseline.py
  run_compare_baselines.py
  requirements.txt
```

## Данные

Используются два основных файла:
- `data/raw/rnc_dataset.csv` — размеченный датасет объявлений
- `data/raw/rnc_mic_key_phrases.csv` — словарь микрокатегорий и характерных фраз

В `rnc_dataset.csv` уже есть разбиение на:
- `train` — для построения и настройки решения
- `val` — для валидации
- `test` — для финальной проверки

## Архитектура эвристического baseline

Основной подход опирается на интерпретируемый pipeline:

1. `data_loader.py`
   Загружает датасет и словарь, приводит списки `mcId` и булевы значения к удобному виду.

2. `text_preprocessing.py`
   Нормализует текст объявления и разбивает его на смысловые блоки.

3. `dictionary_matcher.py`
   Находит кандидатов по `keyPhrases` из словаря микрокатегорий.

4. `split_scorrer.py`
   Считает score самостоятельности услуги по набору положительных и отрицательных маркеров.

5. `draft_generator.py`
   Генерирует текст черновика по шаблону.

6. `predictor.py`
   Собирает все шаги в единый `BaselinePredictor`.

### Почему этот подход выбран как основной

- высокий контроль над решением
- прозрачность и интерпретируемость
- простое масштабирование на новые микрокатегории
- отсутствие зависимости от внешних API и нестабильных моделей
- сильное качество на validation

## Архитектура ML baseline

`src/ml_baseline.py` реализует второй независимый подход:

- текст объявления кодируется через `TF-IDF`
- `OneVsRest Logistic Regression` предсказывает:
  - `detectedMcIds`
  - `splitMcIds`
- отдельная `Logistic Regression` предсказывает `shouldSplit`

ML baseline нужен как исследовательский контрольный эксперимент:
- сравнить rule-based и data-driven подходы
- показать осознанный выбор основного решения

## Результаты

Метрики на validation:

| Baseline | Precision micro | Recall micro | F1 micro | shouldSplit accuracy |
| --- | ---: | ---: | ---: | ---: |
| Heuristic | 0.8960 | 0.9574 | 0.9257 | 0.9642 |
| ML | 0.4294 | 0.8700 | 0.5750 | 0.9485 |

Вывод:
- эвристический baseline существенно превосходит ML baseline по основной multilabel-метрике
- ML-подход был проверен, но не выбран как основной

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

Запуск эвристического baseline:

```bash
python run_baseline.py
```

Что сохраняется:
- `outputs/val_predictions.csv`
- `outputs/test_predictions.csv`
- `outputs/val_predictions.jsonl`
- `outputs/test_predictions.jsonl`

Запуск сравнения heuristic vs ML:

```bash
python run_compare_baselines.py
```

Что сохраняется:
- `outputs/baseline_comparison.csv`

## Dashboard

Для визуального просмотра результатов можно поднять Streamlit dashboard:

```bash
streamlit run dashboard.py
```

В dashboard доступны:
- таблица сравнения heuristic и ML baseline
- bar chart по выбранной метрике
- просмотр предсказаний на validation

## Почему не LLM

Хотя задача допускает LLM-подход, основное решение было оставлено эвристическим по следующим причинам:
- текущий baseline уже показывает высокие метрики
- решение легко контролировать и объяснять
- оно лучше соответствует production-like требованиям задачи
- отсутствуют риски, связанные с внешними API, latency и нестабильностью генерации

LLM можно рассматривать как следующий этап развития:
- для обработки редких неоднозначных кейсов
- для генерации более естественных текстов черновиков

## Команды для защиты

Ключевой тезис проекта:

> Мы сравнили интерпретируемый rule-based подход и классический ML baseline. На validation эвристическое решение показало лучшее качество, при этом сохранив прозрачность, контроль и простоту масштабирования. Поэтому оно выбрано как основное.
