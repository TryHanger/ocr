# Текущее состояние реализации (Implementation State)

Документ отражает **исключительно фактически существующее** состояние кодовой базы и инфраструктуры проекта.

*Дата актуализации: 26 сентября 2026 г.*

---

## 1. Репозиторий и инфраструктура
- **Git-репозиторий:** инициализирован, ветка `main` синхронизирована с `origin https://github.com/TryHanger/ocr.git`.
- **Игнорирование артефактов (`.gitignore`):** настроено исключение виртуальных окружений, кэшей Python/pytest, данных (`data/*`), весов моделей (`models/`, `weights/`), результатов прогонов (`experiments/runs/*`), артефактов coverage (`.coverage*`) и системных файлов.
- **Управляющая и исследовательская документация:**
  - `CLAUDE.md`: протокол управления проектом.
  - `TASKS.md`: трекер задач.
  - `IMPLEMENTATION.md`: фактическое состояние проекта.
  - `DECISIONS.md`: журнал ADR (ADR-001 – ADR-012).
  - `docs/dataset_analysis.md`: сравнительный анализ 4 кандидатов датасетов (SROIE, FUNSD, CORD, DocILE).
  - `docs/dataset_integrity.md`: спецификация предотладочного аудита целостности данных SROIE (чек-лист C-01 – C-10).
  - `docs/research_methodology.md`: **зафиксированный исследовательский стандарт (Frozen Protocol)**: источник SROIE (626 train, 347 test), разбиение на сплиты (train 500/126, test 100/347), двухуровневый протокол OCR (End-to-End и Diagnostic), семантика severity, неизменяемый Raw GT, симметричная нормализация KIE, иерархия препроцессинга B0/B1/B2, RQ1–RQ4, фальсифицируемые гипотезы H1–H4, протокол data leakage и анализ угроз валидности.

---

## 2. Структура каталогов
- `configs/` — YAML-конфигурации:
  - `configs/sroie.yaml`: конфигурация датасета SROIE (канонические параметры, пути, ожидаемые объемы сплитов 626/347, ключи KIE, минимальные размеры).
- `data/` — для локального размещения датасетов и разметки (реальные данные в Git отсутствуют).
- `docs/` — аналитическая, методологическая и аудиторская документация:
  - `docs/dataset_analysis.md`.
  - `docs/dataset_integrity.md`.
  - `docs/research_methodology.md`.
- `experiments/audit/` — каталог для сохранения машиночитаемых отчетов аудита целостности данных (`sroie_integrity.json`).
- `experiments/runs/` — для сохранения метрик и логов (запуски пока не производились).
- `experiments/figures/` — для сохранения графиков.
- `kaggle/` — целевая директория для развертывания ядра Kaggle.
- `scripts/` — директория для CLI утилит запуска:
  - `scripts/audit_sroie.py`: CLI-скрипт сквозного аудита целостности датасета SROIE по чек-листу C-01 – C-10.
- `src/` — корень исходного кода:
  - `src/core/` — **реализован**:
    - `src/core/schemas.py`: DTO и схемы данных (`BoundingBox`, `OCRToken`, `OCRResult`, `KIEResult`, `DocumentMetadata`, `DegradationSpec`, `PreprocessingSpec`, `ExperimentResult`, `OCRGroundTruth`, `KIEGroundTruth`) со строгой валидацией инвариантов и bidirectional JSON/dict сериализацией.
    - `src/core/seed.py`: глобальная фиксация random seed (Python random, NumPy, hash seed).
    - `src/core/contracts.py`: абстрактные базовые классы/контракты будущих модулей (`BaseDatasetAdapter`, `BaseDegradation`, `BasePreprocessor`, `BaseOCREngine`, `BaseKIEEngine`, `BaseEvaluator`).
  - `src/datasets/` — **реализован**:
    - `src/datasets/base.py`: реэкспорт абстрактного контракта `BaseDatasetAdapter`.
    - `src/datasets/sroie.py`: реализация `SROIEAdapter` для канонической структуры Kaggle SROIE v2 (`train/` и `test/` с поддиректориями `img/`, `box/`, `entities/`). Строго соблюдает **Raw Ground Truth Immutability** (без lowercasing, date parsing, float conversion, удаления пунктуации). Детерминированная сортировка, canonical RGB uint8 изображения.
  - `src/degradation/` (реализация алгоритмов отсутствует).
  - `src/preprocessing/` (реализация алгоритмов отсутствует).
  - `src/ocr/` (реализация алгоритмов отсутствует).
  - `src/kie/` (реализация алгоритмов отсутствует).
  - `src/evaluation/` (реализация алгоритмов отсутствует).
  - `src/visualization/` (реализация отсутствует).
- `tests/` — тестовый набор:
  - `tests/fixtures/sroie/` — синтетические фикстуры датасета (валидные, с искажениями OCR, KIE, отсутствующими парами, координатами вне границ кадра).
  - `tests/fixtures/sroie_valid/` — чистые валидные фикстуры для тестирования успешного прохождения аудита.
  - `tests/test_environment.py` — смоук-тест базового окружения.
  - `tests/test_schemas.py` — детальные unit-тесты схем данных, инвариантов, сериализации.
  - `tests/test_seed.py` — unit-тесты детерминизма генераторов случайных чисел.
  - `tests/test_contracts.py` — unit-тесты соблюдения абстрактных контрактов модулей (включая `BaseDatasetAdapter`).
  - `tests/test_dataset_sroie.py` — unit-тесты адаптера `SROIEAdapter` (детерминированность, неизменяемость GT, обработка ошибок, загрузка RGB).
  - `tests/test_audit_sroie.py` — unit-тесты проверок C-01 – C-10 скрипта аудита целостности датасета.

---

## 3. Python-окружение и зависимости
- Конфигурация проекта описана в `pyproject.toml` (современный PEP 621 формат с бэкендом `setuptools`).
- Список базовых зависимостей зафиксирован в `requirements.txt`:
  - `numpy>=1.24.0` (фактически установлен: 2.0.2)
  - `opencv-python>=4.8.0` (фактически установлен: 4.12.0)
  - `PyYAML>=6.0` (фактически установлен: 6.0.3)
  - `pytest>=7.0.0` (фактически установлен: 8.4.2)
- В `pyproject.toml` для разработки подключен:
  - `pytest-cov>=4.0.0` (фактически установлен: 7.1.0)
- Тяжелые ML/OCR/KIE библиотеки (PyTorch, Transformers, PaddleOCR, Tesseract) в проект **не подключались**.

---

## 4. Состояние ML/OCR/KIE модулей
- **Research Protocol & Dataset Strategy:** Полностью зафиксированы и заморожены (**FROZEN**, ADR-008 – ADR-012).
- **Dataset Adapter:** Реализован (`BaseDatasetAdapter`, `SROIEAdapter`).
- **Dataset Integrity Audit:** Реализован (`scripts/audit_sroie.py`, проверки C-01 – C-10).
- **Degradation Engine:** Не реализован (контракт описан через `BaseDegradation` и `DegradationSpec`).
- **Preprocessing Pipeline:** Не реализован (контракт описан через `BasePreprocessor` и `PreprocessingSpec`).
- **OCR Engine Abstraction & Models:** Не реализованы (контракт описан через `BaseOCREngine`, `OCRToken`, `OCRResult`).
- **KIE Extractor:** Не реализован (контракт описан через `BaseKIEEngine` и `KIEResult`).
- **Evaluation & Metrics:** Не реализованы (контракт описан через `BaseEvaluator` и `ExperimentResult`).
- **Experiment Runner:** Не реализован.
- **Kaggle Automation Scripts:** Не реализованы.

---

## 5. Тестирование и покрытие
- Команда запуска тестов:
  ```bash
  pytest --cov=src/core --cov=src/datasets --cov=scripts.audit_sroie --cov-report=term-missing
  ```
- Результат: **206 тестов пройдено успешно**, покрытие:
  - `src/core`: **100%** (374 statements, 0 missed)
  - `src/datasets`: **100%** (155 statements, 0 missed)
  - `scripts/audit_sroie.py`: **92%** (290 statements, 22 missed)
  - Суммарное покрытие: **97%** (819 statements, 22 missed).

