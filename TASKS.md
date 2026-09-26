# TASKS

## Active

*(Нет активных задач)*

## Completed

- **TASK-004 — SROIE Dataset Adapter + Dataset Integrity Audit**
  - *Дата завершения:* 2026-09-26
  - *Статус:* Завершена успешно.
  - *Результат:* Создан пакет `src/datasets/`, реализованы `BaseDatasetAdapter` и `SROIEAdapter` (`src/datasets/sroie.py`) с полной поддержкой канонической структуры Kaggle SROIE v2 (`train/` и `test/` с подкаталогами `img/`, `box/`, `entities/`). Реализован строгий принцип Raw Ground Truth Immutability (нет lowercasing, date parsing, float conversion, удаления пунктуации). Создан CLI-скрипт `scripts/audit_sroie.py` для валидации чек-листа C-01 – C-10 из `docs/dataset_integrity.md`, формирующий машиночитаемый JSON-отчет (`experiments/audit/sroie_integrity.json`) и консольную сводку с корректным exit code. Созданы синтетические фикстуры `tests/fixtures/sroie/` (валидные, с дефектами OCR/KIE, непарными файлами, out-of-bounds координатами) и чистые фикстуры `tests/fixtures/sroie_valid/`. Добавлены unit-тесты адаптера и аудита, все 206 тестов проекта пройдены со 100% покрытием `src/core` и `src/datasets` (97% суммарно).


- **TASK-003-R2 — Research Freeze**
  - *Дата завершения:* 2026-09-26
  - *Статус:* Завершена успешно. Исследовательский протокол **FROZEN**.
  - *Результат:* Зафиксирован единый Source of Truth датасета SROIE (`urbikn/sroie-datasetv2`, 626 train, 347 test). Детально объяснено происхождение расхождения 347 vs 361 (14 документов Task 1/2 без Task 3 KIE GT). Утверждена схема сплитов (train 500/126, test 100/347). Численные параметры деградаций вынесены из методологии в процедуру будущей калибровки на validation-сплите. Гипотезы H1–H4 очищены от предвзятых численных порогов. Зафиксированы метрики OCR/KIE, протоколы reading order, краевые случаи, симметричная нормализация и failure modes. Создан аудит-документ `docs/dataset_integrity.md`, обновлены `docs/research_methodology.md`, `DECISIONS.md` (ADR-012), `IMPLEMENTATION.md`.

- **TASK-003-R1 — Research Validation: Dataset и методология OCR/KIE Robustness**
  - *Дата завершения:* 2026-09-26
  - *Статус:* Завершена успешно.
  - *Результат:* Проведена строгая валидация по первоисточникам для SROIE, FUNSD, CORD, DocILE. Исправлено и формализовано определение `severity=0` (контрольный скан из датасета с естественными артефактами, а не идеальный документ). Зафиксирован двухуровневый протокол OCR (End-to-End как Primary, GT-Region как Diagnostic). Утверждена политика неизменяемого Raw GT для OCR и симметричной нормализации для KIE. Сформулированы RQ1–RQ4, гипотезы H1–H4, протокол data leakage и анализ угроз валидности (Threats to Validity) в документе `docs/research_methodology.md`. Добавлены `ADR-009`, `ADR-010`, `ADR-011`.

- **TASK-003 — Dataset Strategy: анализ и выбор датасета для OCR/KIE**
  - *Дата завершения:* 2026-09-26
  - *Статус:* Завершена успешно.
  - *Результат:* Проведён предварительный анализ датасетов SROIE, FUNSD, CORD, WildReceipt. Подготовлен первичный документ `docs/dataset_analysis.md`. Выбран базовый датасет SROIE. Спроектирована архитектура `BaseDatasetAdapter`. Принят ADR-008.

- **TASK-002 — Core Schema, DTO и контракты модулей**
  - *Дата завершения:* 2026-09-26
  - *Статус:* Завершена успешно.
  - *Результат:* Создан пакет `src/core/`, реализованы все DTO (`BoundingBox`, `OCRToken`, `OCRResult`, `KIEResult`, `DocumentMetadata`, `DegradationSpec`, `PreprocessingSpec`, `ExperimentResult`, `OCRGroundTruth`, `KIEGroundTruth`), двусторонняя сериализация/десериализация `to_dict()`/`from_dict()`, строгая валидация инвариантов, абстрактные контракты модулей (`BaseDegradation`, `BasePreprocessor`, `BaseOCREngine`, `BaseKIEEngine`, `BaseEvaluator`), утилита детерминированного сида `set_seed()`. Покрытие тестами модуля `src/core` составляет 100% (161 тест).

- **TASK-001 — Инициализация репозитория, Git, управляющей документации и базового окружения**
  - *Дата завершения:* 2026-09-26
  - *Статус:* Завершена успешно.
  - *Результат:* Инициализирован Git, настроен `.gitignore`, созданы каталоги проекта, `CLAUDE.md`, `TASKS.md`, `IMPLEMENTATION.md`, `DECISIONS.md`, `pyproject.toml`, `requirements.txt`, `README.md`, смоук-тесты в `tests/test_environment.py` успешно пройдены.
