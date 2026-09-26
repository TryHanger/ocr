# TASKS

## Active

*(Нет активных задач)*

- **TASK-010 — Unified Experiment Runner & Experiment Matrix**
  - *Дата завершения:* 2026-09-26 (с учетом уточнений TASK-010-R1)
  - *Статус:* Завершена успешно (PASS).
  - *Результат:* Реализован единый детерминированный исследовательский раннер `UnifiedExperimentRunner` (`src/experiments/`) и CLI точка входа `scripts/run_experiment.py` для проведения экспериментов по устойчивости OCR и downstream KIE к синтетическим деградациям и влияния препроцессинга ($B_0, B_1, B_2$):
    1. *Канонический конвейер со строгой изоляцией:* Реализована строгая последовательность шагов `SROIE document → split → degradation → preprocessing → OCR → KIE → GT loading → evaluation → artifacts`. OCR и KIE движки никогда не получают Ground Truth данные (ни боксы, ни транскрипции, ни сущности, ни адаптер датасета).
    2. *Семантика экспериментальных условий:*
       - Ровно одно контрольное условие: `D0_S0_P0` (`baseline_type="B0"`).
       - $B_1$ (только деградация): $D_1..D_8 \times \text{severity} \{1..4\} \times P_0$.
       - $B_2$ (деградация + препроцессинг): $D_1..D_8 \times \text{severity} \{1..4\} \times P^*$.
       - Severity 0 для деградаций $D_1..D_8$ строго запрещен (валидация инварианта в `ExperimentCondition`).
       - Минимальная CPU smoke-матрица: `D0_S0_P0`, `D1_S1_P0`, `D1_S4_P0`, `D1_S1_P_clahe`, `D1_S4_P_clahe`.
    3. *Защита тестового сплита и политика фикстур:*
       - Запрос сплита `test` ($N=347$) падает с громкой ошибкой, если не передан флаг `--allow-test`.
       - При отсутствии реальных данных SROIE запуск без флага `--allow-fixture` падает с ошибкой `REAL_DATA_REQUIRED`. Фикстуры никогда не подменяют реальные данные silently; статус данных явно фиксируется как `PENDING_REAL_DATA`.
    4. *Детерминированный вывод сидов:*
       - Реализована функция `derive_seed(experiment_seed, doc_id, deg_type, severity, prep_id)` на базе SHA-256 с гарантией ортогональности и независимости сидов.
    5. *Статистика и bootstrap:*
       - Реализован расчет доверительных интервалов bootstrap 95% CI (1000 итераций) на уровне документов для всех OCR и KIE метрик (mean, median, std, ci_95) с фиксированным сидом (`seed + 10000`).
    6. *Учет сбоев (Failure Accounting):*
       - Ошибки OCR, KIE и evaluation фиксируются в поле `status` (`ocr_failed`, `kie_failed`, `evaluation_failed`) с сохранением типа и текста исключения без silent drop документов.
    7. *Артефакты и воспроизводимость:*
       - Директория прогона `experiments/runs/<experiment_id>/` содержит `config.yaml`, `manifest.json` (с config hash, git commit, runtime metadata, data_status), `per_document.jsonl`, `summary.json`, `logs/run.log`.
    8. *Тестирование:*
       - Добавлены тесты `tests/test_experiment_runner.py` (13 тестов, 83% покрытие модуля `src/experiments`). Все 391 тест проекта пройдены со 100% успехом и суммарным покрытием 92%.
       - Проведен контролируемый smoke-прогон на Development-выборке (`smoke_run --allow-fixture`), статус зафиксирован как `PENDING_REAL_DATA` (инженерная верификация, без научных утверждений).

- **TASK-009 — KIE Baseline Engine + Evaluation**
  - *Дата завершения:* 2026-09-26 (с учетом уточнений TASK-009-R1)
  - *Статус:* Завершена успешно (PASS).
  - *Результат:* Реализован детерминированный правилосодержащий baseline для downstream-задачи извлечения ключевой информации (Key Information Extraction, KIE) поверх ранее зафиксированного OCR-стека (RapidOCR + ONNX Runtime + PP-OCRv6 small):
    1. *Контракт и Zero GT Leakage:* Обновлен абстрактный контракт `BaseKIEEngine` в `src/core/contracts.py`: сигнатура строго зафиксирована как `extract(self, ocr_result: OCRResult, document_id: str) -> KIEResult`. Аргумент `image` полностью исключен из production API. На вход KIE передается исключительно DTO `OCRResult` и идентификатор документа; никакие эталонные данные (GT boxes, transcripts, entity labels, dataset adapter) не передаются и недоступны движку.
    2. *Модуль `src/kie/`:* Создана модульная архитектура правил в `src/kie/rules/` (`company`, `date`, `address`, `total`) с генерацией кандидатов `FieldCandidate`, вычислением нормализованных уровней уверенности `[0.0, 1.0]` и сохранением аудиторского следа происхождения токенов (`metadata["field_provenance"]`). Реализован `RuleBasedKIEEngine`, детерминированный `MockKIEEngine` и фабрика `get_kie_engine()`. Эвристики ориентированы на структуру квитанций SROIE и задокументированы как предварительные (provisional) без оверфиттинга под конкретные документы.
    3. *Политика агрессивного фоллбэка:* Эвристика «наибольшее число = total» по умолчанию строго отключена (`fallback_largest_amount: false`). При явной активации в конфигурации кандидату назначается штраф к уверенности (`fallback_penalty: 0.5`).
    4. *Двухуровневая система метрик (`src/evaluation/kie_metrics.py`, `KIEEvaluator`):* Структурно разделены:
       - `analytical_metrics`: пополевые Precision, Recall, F1 (как сырые `raw`, так и нормализованные `normalized`), Macro Precision, Recall, F1 по 4 полям, независимые показатели `raw_doc_em` и `normalized_doc_em`.
       - `sroie_official_compatible`: микро-метрики Precision, Recall и Hmean (F1) на уровне сущностей согласно официальному протоколу ICDAR SROIE Task 3.
       - Нормализаторы: `normalize_kie_text` (NFC, lowercase, схлопывание пробелов, удаление граничной пунктуации с сохранением внутренней) и `normalize_total_amount` (удаление валютных префиксов `$`, `RM`, `MYR` с сохранением десятичных дробей).
    5. *Сквозной раннер и аудит исполнения:* Реализован CLI-раннер `scripts/run_kie_baseline.py` с принудительной последовательностью фаз: `ocr_inference` $\to$ `kie_inference` $\to$ `gt_loading` $\to$ `evaluation`. Результаты сохраняются в `experiments/runs/kie_baseline_report.json` со статусом `PENDING_REAL_DATA`.
    6. *Конфигурация и документация:* Создан конфигурационный файл `configs/kie.yaml`, принят `ADR-016` в `DECISIONS.md`, обновлены `IMPLEMENTATION.md` и управляющая документация.
    7. *Тестирование:* Добавлены исчерпывающие тестовые наборы `tests/test_kie.py`, `tests/test_kie_metrics.py`, `tests/test_kie_runner.py`. Все 378 тестов проекта успешно пройдены со 100% успехом и суммарным покрытием 94%.

- **TASK-008 — OCR Stack Refresh & Reproducibility Lock**
  - *Дата завершения:* 2026-09-26 (исправления TASK-008-R1 приняты)
  - *Статус:* Завершена успешно (PASS).
  - *Результат:* Базовый OCR-стек успешно обновлен с устаревшей версии PP-OCRv4 на современный стек RapidOCR (`rapidocr==3.9.2`) + ONNX Runtime (`onnxruntime==1.30.0`) с моделями PP-OCRv6 (small tier) для детекции (`PP-OCRv6_det_small.onnx`) и распознавания (`PP-OCRv6_rec_small.onnx`), а также мобильным классификатором ориентации (`ch_ppocr_mobile_v2.0_cls_mobile.onnx`). Реализован механизм криптографической фиксации воспроизводимости (Reproducibility Lock) с динамической интроспекцией сессий ONNX Runtime и проверкой SHA256 хешей файлов весов (`verify_model_stack()`). Эксплицитно задокументирована зависимость разрешения путей от приватного атрибута `_model_path` обертки RapidOCR с гарантией аварийного падения с громкой ошибкой (fail loudly) в случае его исчезновения. Реализован экспорт структурированного манифеста `experiments/runs/ocr_stack_manifest.json` с четким разделением на секции `configured`, `resolved`, `environment`, а также валидированный блок `resize_policy` (`configured`, `resolved`, `verified: true`). Проведен строгий аудит семантики билинейного ресайза RapidOCR (`max_side_len: 2000`, `min_side_len: 30`, `det_limit_side_len: 736`, `det_limit_type: min`) и зафиксировано строгое методологическое правило: базисы $B_0, B_1, B_2$ используют одну и ту же политику и конфигурацию OCR-ресайза, а физические результирующие размеры могут легитимно различаться из-за различий входных изображений. Строгий контракт `BaseOCREngine` (`recognize(image, document_id) -> OCRResult`) и принцип Strict GT Isolation полностью сохранены. Обновлены конфигурация `configs/ocr.yaml`, зависимости в `requirements.txt` и `pyproject.toml`, скрипт `scripts/run_ocr_baseline.py`. Статус бенчмарка реальных данных строго зафиксирован как `REAL-DATA OCR BASELINE: PENDING`. Добавлены критические unit- и регрессионные тесты в `tests/test_ocr_stack.py` (14 тестов, включая loud failure при потере `_model_path`, несовпадение хешей, отсутствующие файлы, совместимость с legacy-выходом). Все 351 тест проекта пройден со 100% успехом и общим покрытием 94%. Принят обновленный `ADR-015`.

- **TASK-007 — Preprocessing Engine + B0/B1/B2 Baselines**
  - *Дата завершения:* 2026-09-26
  - *Статус:* Завершена успешно (READY FOR ARCHITECT REVIEW).
  - *Результат:* Реализован модуль препроцессинга `src/preprocessing/` на базе абстрактного контракта `BasePreprocessor` и DTO `PreprocessingSpec`. Реализованы 5 независимых детерминированных примитивов: P1 (Grayscale: преобразование RGB/RGBA в 2D uint8), P2 (Denoising: median, bilateral, FastNlMeans), P3 (CLAHE: локальное выравнивание гистограммы для 2D grayscale или L-канала CIE LAB для RGB), P4 (Adaptive Binarization: Gaussian, Mean, Otsu со строгим выходом $\{0, 255\}$ uint8), P5 (Deskew: оценка угла текста по Canny + HoughLinesP, поворот с белыми полями и ограничением $\pm 15^\circ$, zero GT access). Реализована последовательная композиция `PreprocessingPipeline` и реестр-фабрика `get_preprocessor()`. Реализован раннер аналитических базисов `run_b0_b1_b2_comparison` для оценки веток $B_0$ (original $\to$ OCR), $B_1$ (degraded $\to$ OCR), $B_2$ (degraded $\to$ preprocessing $\to$ OCR) с полной изоляцией метрик от препроцессинга. Создан конфигурационный файл `configs/preprocessing.yaml` с кандидатными сетками и предварительными (provisional) параметрами. Реализован CLI-скрипт `scripts/run_preprocessing_baseline.py` с фиксацией отчета `experiments/runs/preprocessing_baseline_report.json` со статусом `real_data_preprocessing_baseline: "PENDING"`. Принят `ADR-014`. Написан исчерпывающий набор unit-тестов `tests/test_preprocessing.py` (31 тест). Все 337 тестов проекта пройдены с общим покрытием 94%.

- **TASK-006 — OCR Engine Baseline + End-to-End OCR Evaluation**
  - *Дата завершения:* 2026-09-26
  - *Статус:* Завершена успешно (READY FOR ARCHITECT REVIEW).
  - *Результат:* Реализован базовый локальный OCR-движок на базе RapidOCR (`rapidocr-onnxruntime==1.4.4`, `onnxruntime==1.30.0`, PP-OCRv4 ONNX модели) и абстракции `BaseOCREnginePrimitive` со строгой изоляцией от Ground Truth в основном контракте `recognize(image, document_id) -> OCRResult`. Реализована детерминированная эвристика порядка чтения строк (`sort_tokens_reading_order` с настраиваемым допуском `line_tolerance_factor=0.5`). Реализован отдельный диагностический метод `diagnostic_gt_region_recognition` для анализа детекции/распознавания. Создан пакет `src/evaluation/ocr_metrics.py` с реализацией расстояния Левенштейна, CER, WER и строго определенного **Character-NED similarity** ($1.0 - \frac{\text{edit\_distance}}{\max(\text{len}(pred), \text{len}(gt), 1)}$). Реализована симметричная нормализация текста OCR (Unicode NFC, схлопывание пробелов, удаление управляющих символов), сохраняющая 4 текстовых представления (`raw_pred`, `raw_gt`, `norm_pred`, `norm_gt`). Создана конфигурация `configs/ocr.yaml` и CLI runner `scripts/run_ocr_baseline.py`. В отсутствие скачанного датасета сформирован отчет `experiments/runs/ocr_baseline_report.json` со статусом `real_data_ocr_baseline: "PENDING"`. Принят `ADR-013`. Добавлены исчерпывающие unit-тесты (`tests/test_ocr.py`, `tests/test_ocr_metrics.py`). Все 306 тестов проекта пройдены с общим покрытием 96% (100% evaluation, 100% core, 100% datasets, 98% degradation, 91% ocr).

- **TASK-005 — Degradation Engine + Severity Calibration**
  - *Дата завершения:* 2026-09-26
  - *Статус:* Завершена успешно (реализационная часть модуля деградаций и калибровочного пайплайна).
  - *Результат:* Создан пакет `src/degradation/`, реализованы 8 детерминированных примитивов деградаций (D1: Gaussian Blur, D2: Motion Blur, D3: Gaussian Noise, D4: JPEG Compression, D5: Downsampling, D6: Rotation, D7: Perspective Distortion, D8: Shadow) на базе `BaseDegradationPrimitive`. Реализован строгий контракт severity 0 (bitwise identity pass-through, `image.copy()`, zero mutation) и неизменяемость входных данных. Создана абстракция `DegradationPipeline` и фабрика `get_degradation()`. Создан конфигурационный файл `configs/degradation.yaml` с кандидатными сетками и предварительными параметрами. Реализован модуль калибровки `scripts/calibrate_degradations.py`, вычисляющий объективные метрики искажения (PSNR, SSIM, MAE, Laplacian ratio, Luminance drop), проверяющий монотонность физических параметров и метрик ($1 < 2 < 3 < 4$). Сгенерированы артефакты `experiments/calibration/` (`calibration_config.yaml`, `calibration_report.json`, визуальные образцы `samples/`). Текущие параметры зафиксированы как стартовый предварительный набор (**provisional / fixture-based**), который послужит отправной точкой; статус калибровки на реальных данных явно зафиксирован как `real_data_calibration: PENDING` (`calibration_dataset: synthetic_fixture`, `research_validation_size: 126`, `final_calibration_completed: false`). Окончательная калибровка на 126 документах validation split будет проведена после размещения реального датасета в `data/`. Написан исчерпывающий набор unit-тестов `tests/test_degradation.py`, все 270 тестов проекта пройдены с общим покрытием 96% (100% core, 100% datasets, 98% degradation).

- **TASK-004-R1 — Correct SROIE Adapter Raw-GT Semantics**
  - *Дата завершения:* 2026-09-26
  - *Статус:* Завершена успешно (READY FOR ARCHITECT REVIEW).
  - *Результат:* Исправлена семантика Raw Ground Truth: удален clipping координат в `SROIEAdapter.get_ocr_ground_truth` и ограничение в `BoundingBox`; координаты разметки сохраняются 1-в-1 с источником, включая отрицательные и выходящие за пределы кадра. Восстановлена строгая валидация формата OCR: ровно 8 координат, парсинг гарантирует отказ при < 8 и > 8 координатах без поглощения чисел в текст. Удален несанкционированный порог площади $H \times W \ge 10000$ (C-03 теперь требует исключительно $H > 0$ и $W > 0$). Отчет аудита дополнен метаданными `"mode": "synthetic_fixture"` и `"source": "tests/fixtures/sroie"`, а также поддержкой строгого режима сплитов `--strict` / `--strict-counts`. Все 208 тестов пройдены со 100% покрытием `src/core` и `src/datasets` (97% суммарно).

- **TASK-004 — SROIE Dataset Adapter + Dataset Integrity Audit**
  - *Дата завершения:* 2026-09-26
  - *Статус:* Доработана в рамках TASK-004-R1 (READY FOR ARCHITECT REVIEW).
  - *Результат:* Создан пакет `src/datasets/`, реализованы `BaseDatasetAdapter` и `SROIEAdapter` (`src/datasets/sroie.py`) с полной поддержкой канонической структуры Kaggle SROIE v2 (`train/` и `test/` с подкаталогами `img/`, `box/`, `entities/`). Реализован строгий принцип Raw Ground Truth Immutability (нет lowercasing, date parsing, float conversion, удаления пунктуации). Создан CLI-скрипт `scripts/audit_sroie.py` для валидации чек-листа C-01 – C-10 из `docs/dataset_integrity.md`, формирующий машиночитаемый JSON-отчет (`experiments/audit/sroie_integrity.json`) и консольную сводку с корректным exit code. Созданы синтетические фикстуры `tests/fixtures/sroie/` (валидные, с дефектами OCR/KIE, непарными файлами, out-of-bounds координатами) и чистые фикстуры `tests/fixtures/sroie_valid/`. Добавлены unit-тесты адаптера и аудита.


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
