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
  - `configs/degradation.yaml`: конфигурация примитивов деградаций D1–D8, сетки параметров кандидатов, предварительные калибровочные параметры (provisional / fixture-based) для severities 1..4 и сиды.
  - `configs/ocr.yaml`: конфигурация базового OCR-движка RapidOCR, моделей ONNX PP-OCRv4, параметров порядка чтения (reading order tolerance) и метрик.
  - `configs/preprocessing.yaml`: конфигурация примитивов предобработки P1–P5, кандидатных сеток параметров, предварительных (provisional) настроек и пайплайнов для baselines.
- `data/` — для локального размещения датасетов и разметки (реальные данные в Git отсутствуют).
- `docs/` — аналитическая, методологическая и аудиторская документация:
  - `docs/dataset_analysis.md`.
  - `docs/dataset_integrity.md`.
  - `docs/research_methodology.md`.
- `experiments/audit/` — каталог для сохранения машиночитаемых отчетов аудита целостности данных (`sroie_integrity.json`).
- `experiments/calibration/` — артефакты калибровки уровней severity 1..4:
  - `experiments/calibration/calibration_config.yaml`: параметры severity 1..4 для каждого типа (предварительный стартовый набор provisional / fixture-based).
  - `experiments/calibration/calibration_report.json`: полный машиночитаемый отчет с метриками (PSNR, SSIM, MAE, Laplacian, Luminance) и обоснованием выбора; статус зафиксирован как `real_data_calibration: PENDING`, `calibration_dataset: synthetic_fixture`, `research_validation_size: 126`, `final_calibration_completed: false`.
  - `experiments/calibration/samples/`: визуальные образцы каждого уровня деградации (severities 0..4).
- `experiments/runs/` — артефакты прогонов:
  - `experiments/runs/ocr_baseline_report.json`: отчет прогона базового OCR пайплайна со статусом `real_data_ocr_baseline: "PENDING"`.
  - `experiments/runs/preprocessing_baseline_report.json`: отчет прогона B0/B1/B2 препроцессинга со статусом `real_data_preprocessing_baseline: "PENDING"`.
- `experiments/figures/` — для сохранения графиков.
- `kaggle/` — целевая директория для развертывания ядра Kaggle.
- `scripts/` — директория для CLI утилит запуска:
  - `scripts/audit_sroie.py`: CLI-скрипт сквозного аудита целостности датасета SROIE по чек-листу C-01 – C-10.
  - `scripts/calibrate_degradations.py`: CLI-скрипт калибровки параметров деградаций на validation-выборке и генерации калибровочных артефактов.
  - `scripts/run_ocr_baseline.py`: CLI-скрипт сквозного прогона OCR-пайплайна на SROIE с замером CER, WER и Character-NED similarity.
  - `scripts/run_preprocessing_baseline.py`: CLI-скрипт сравнительного прогона аналитических базисов B0, B1, B2 на SROIE.
- `src/` — корень исходного кода:
  - `src/core/` — **реализован**:
    - `src/core/schemas.py`: DTO и схемы данных (`BoundingBox`, `OCRToken`, `OCRResult`, `KIEResult`, `DocumentMetadata`, `DegradationSpec`, `PreprocessingSpec`, `ExperimentResult`, `OCRGroundTruth`, `KIEGroundTruth`) со строгой валидацией инвариантов и bidirectional JSON/dict сериализацией. `BoundingBox` разрешает координаты вне границ изображения и отрицательные координаты, сохраняя геометрию первоисточника без искажений.
    - `src/core/seed.py`: глобальная фиксация random seed (Python random, NumPy, hash seed).
    - `src/core/contracts.py`: абстрактные базовые классы/контракты модулей (`BaseDatasetAdapter`, `BaseDegradation`, `BasePreprocessor`, `BaseOCREngine`, `BaseKIEEngine`, `BaseEvaluator`).
  - `src/datasets/` — **реализован**:
    - `src/datasets/base.py`: реэкспорт абстрактного контракта `BaseDatasetAdapter`.
    - `src/datasets/sroie.py`: реализация `SROIEAdapter` для канонической структуры Kaggle SROIE v2 (`train/` и `test/` с поддиректориями `img/`, `box/`, `entities/`). Строго соблюдает **Raw Ground Truth Immutability**: координаты не обрезаются и не нормализуются (`source == adapter`), полигоны и derived `BoundingBox` сохраняются verbatim, валидация формата OCR требует ровно 8 координат, текст транскрипции не модифицируется (без lowercasing, date parsing, float conversion, удаления пунктуации). Детерминированная сортировка, canonical RGB uint8 изображения.
  - `src/degradation/` — **реализован**:
    - `src/degradation/base.py`: абстрактный класс `BaseDegradationPrimitive` со сквозной валидацией входного изображения, строгим соблюдением инварианта `severity = 0` (identity pass-through, `image.copy()`, zero mutation) и контролем типа uint8 и размерностей выхода.
    - `src/degradation/gaussian_blur.py`: примитив D1 (Gaussian Blur, параметры `sigma`, `kernel_size`).
    - `src/degradation/motion_blur.py`: примитив D2 (Motion Blur, параметры `kernel_length`, `angle`).
    - `src/degradation/gaussian_noise.py`: примитив D3 (Gaussian Noise, параметры `mean`, `std`, детерминированный RNG по `spec.seed`).
    - `src/degradation/jpeg_compression.py`: примитив D4 (In-memory JPEG Compression, параметр `quality` 1..100, сохранение цветового пространства RGB).
    - `src/degradation/downsampling.py`: примитив D5 (Downsampling, параметры `scale_factor`/`factor`, `cv2.INTER_AREA` -> `cv2.INTER_LINEAR`, восстановление исходных габаритов).
    - `src/degradation/rotation.py`: примитив D6 (Geometric Rotation, параметры `angle`, `border_mode`, сохранение габаритов кадра, заливка белым фоном).
    - `src/degradation/perspective.py`: примитив D7 (Four-point Perspective, параметр `distortion_scale`, детерминированное смещение 4 углов по `spec.seed`).
    - `src/degradation/shadow.py`: примитив D8 (Synthetic Shadow, параметры `opacity`, `angle`, `coverage`, сигмоидный градиент затенения).
    - `src/degradation/pipeline.py`: класс композиции `DegradationPipeline` и фабрика `get_degradation()`.
  - `src/ocr/` — **реализован**:
    - `src/ocr/base.py`: базовый класс `BaseOCREnginePrimitive`, реализующий абстрактный контракт `BaseOCREngine`. Строгая изоляция Ground Truth: метод `recognize(image, document_id) -> OCRResult` принимает строго `(image, document_id)`. Детерминированная эвристика порядка чтения `sort_tokens_reading_order` с настраиваемым параметром `line_tolerance_factor`. Отдельный изолированный диагностический метод `diagnostic_gt_region_recognition(image, document_id, gt_boxes)`.
    - `src/ocr/rapid_ocr.py`: реализация `RapidOCREngine` на базе ONNX Runtime и моделей PP-OCRv4. Корректная обработка пустых/однородных изображений, сохранение 4-точечных ориентированных полигонов в `metadata["raw_quadrilaterals"]` и derived `BoundingBox` без клиппинга.
    - `src/ocr/factory.py`: фабрика `get_ocr_engine(name, params)` и реализация `MockOCREngine` для быстрых детерминированных тестов.
    - `src/ocr/__init__.py`: публичные экспорты модуля.
  - `src/evaluation/` — **реализован (OCR метрики)**:
    - `src/evaluation/ocr_metrics.py`: реализация расстояния Левенштейна, CER, WER и строго определенного **Character-NED similarity** ($1.0 - \frac{\text{edit\_distance}}{\max(\text{len}(pred), \text{len}(gt), 1)}$). Функция симметричной нормализации `normalize_ocr_text` (Unicode NFC, схлопывание пробелов, удаление управляющих символов). DTO `OCREvaluationResult` с сохранением 4 представлений текста (`raw_pred`, `raw_gt`, `norm_pred`, `norm_gt`) и сырых/нормализованных метрик.
    - `src/evaluation/__init__.py`: публичные экспорты модуля.
  - `src/preprocessing/` — **реализован**:
    - `src/preprocessing/base.py`: абстрактный класс `BasePreprocessingPrimitive` на базе контракта `BasePreprocessor`. Защита от мутаций входа (`image.copy()`), валидация uint8/2D/3D, pass-through для `spec.enabled == False`. Допускает изменение каналов (например, RGB -> 2D grayscale/binary).
    - `src/preprocessing/grayscale.py`: примитив P1 (Grayscale, преобразование RGB/RGBA в 2D uint8, сквозной пропуск для 2D).
    - `src/preprocessing/denoise.py`: примитив P2 (Denoising, поддержка median, bilateral, FastNlMeans, сохранение размерностей).
    - `src/preprocessing/clahe.py`: примитив P3 (CLAHE, прямое применение к 2D grayscale или к L-каналу CIE LAB для RGB без сдвига цветов).
    - `src/preprocessing/binarization.py`: примитив P4 (Adaptive Binarization, Gaussian, Mean, Otsu, выход строго $\{0, 255\}$ uint8).
    - `src/preprocessing/deskew.py`: примитив P5 (Deskew, детекция доминирующего угла текста через Canny + HoughLinesP, поворот с белыми полями, ограничение max_angle 15.0).
    - `src/preprocessing/pipeline.py`: класс последовательной композиции `PreprocessingPipeline` и фабрика `get_preprocessor()`.
    - `src/preprocessing/baseline.py`: бейслайн-раннер `run_b0_b1_b2_comparison` для сквозного сравнительного выполнения B0, B1, B2.
    - `src/preprocessing/__init__.py`: публичные экспорты модуля.
  - `src/kie/` (реализация алгоритмов отсутствует).
  - `src/visualization/` (реализация отсутствует).
- `tests/` — тестовый набор:
  - `tests/fixtures/sroie/` — синтетические фикстуры датасета (валидные, с дефектами OCR, KIE, отсутствующими парами, координатами вне границ кадра).
  - `tests/fixtures/sroie_valid/` — чистые валидные фикстуры для тестирования успешного прохождения аудита.
  - `tests/test_environment.py` — смоук-тест базового окружения.
  - `tests/test_schemas.py` — детальные unit-тесты схем данных, инвариантов, сериализации.
  - `tests/test_seed.py` — unit-тесты детерминизма генераторов случайных чисел.
  - `tests/test_contracts.py` — unit-тесты соблюдения абстрактных контрактов модулей.
  - `tests/test_dataset_sroie.py` — unit-тесты адаптера `SROIEAdapter` (детерминированность, неизменяемость GT, обработка ошибок, загрузка RGB, сохранение необрезанных координат, строгая проверка 8 координат).
  - `tests/test_audit_sroie.py` — unit-тесты проверок C-01 – C-10 скрипта аудита целостности датасета.
  - `tests/test_degradation.py` — исчерпывающие unit-тесты для 8 примитивов деградаций (severity 0 identity, неизменяемость входа, uint8/shape contracts, детерминизм, стохастические сиды, монотонность, пайплайн, фабрика, калибровочные метрики и CLI).
    - `tests/test_ocr_metrics.py` — unit-тесты метрик CER, WER, Character-NED similarity (граничные случаи, edge-cases, Unicode NFC, нормализация).
    - `tests/test_ocr.py` — unit-тесты контрактов OCR, входной валидации, строгого отсутствия утечек GT в primary API, порядка чтения, инференса RapidOCR, пустого изображения, диагностического режима и runner CLI.
    - `tests/test_ocr_stack.py` — исчерпывающие unit- и интеграционные тесты стека OCR: верификация моделей PP-OCRv6, проверка криптографических SHA256 хешей, интроспекция манифеста (configured, resolved, environment), поддержка RGB/Grayscale/Binary, аудит политики ресайза, детерминизм инференса и тайминги на CPU.
    - `tests/test_preprocessing.py` — unit-тесты контрактов препроцессинга, P1–P5 примитивов, неизменяемости входа, детерминизма, отсутствия утечек GT, пайплайна, фабрики и аналитических базисов B0/B1/B2.

---

## 3. Python-окружение и зависимости
- Конфигурация проекта описана в `pyproject.toml` (современный PEP 621 формат с бэкендом `setuptools`).
- Список базовых зависимостей зафиксирован в `requirements.txt`:
  - `numpy>=1.24.0` (фактически установлен: 2.0.2)
  - `opencv-python>=4.8.0` (фактически установлен: 4.12.0)
  - `PyYAML>=6.0` (фактически установлен: 6.0.3)
  - `pytest>=7.0.0` (фактически установлен: 8.4.2)
  - `rapidocr==3.9.2` (зафиксирован точно, PP-OCRv6 small models bundled)
  - `onnxruntime==1.30.0` (зафиксирован точно, CPUExecutionProvider locked)
- В `pyproject.toml` для разработки подключен:
  - `pytest-cov>=4.0.0` (фактически установлен: 7.1.0)
- Тяжелые ML/OCR фреймворки с внешними компилируемыми бинарниками или CUDA-зависимостями (PyTorch, Transformers, PaddlePaddle, Tesseract) в проект **не подключались**.

---

## 4. Состояние ML/OCR/KIE модулей
- **Research Protocol & Dataset Strategy:** Полностью зафиксированы и заморожены (**FROZEN**, ADR-008 – ADR-015).
- **Dataset Adapter:** Реализован (`BaseDatasetAdapter`, `SROIEAdapter`) в полном соответствии с raw-GT семантикой TASK-004-R1 (сохранение геометрии без клиппинга, строгая проверка 8 координат, неизменяемость транскрипций).
- **Dataset Integrity Audit:** Реализован (`scripts/audit_sroie.py`, проверки C-01 – C-10, поддержка аргументов `--mode` и `--strict`, очистка C-03 от порогов площади, явная фиксация режима фикстур в отчете).
- **Degradation Engine:** Реализован (`src/degradation/`, примитивы D1–D8, `DegradationPipeline`, `get_degradation()`, конфигурация `configs/degradation.yaml`, скрипт калибровки `scripts/calibrate_degradations.py`, калибровочные артефакты в `experiments/calibration/`). Полный детерминизм, строгий инвариант severity 0 identity, неизменяемость входов, сохранение размерностей и типа uint8. Текущие параметры зафиксированы как стартовый предварительный набор (**provisional / fixture-based**); окончательная исследовательская калибровка на валидационном сплите SROIE ($N=126$) ожидает появления датасета (`real_data_calibration: PENDING`, `calibration_dataset: synthetic_fixture`, `final_calibration_completed: false`).
- **OCR Engine Baseline & Reproducibility Lock:** Актуализирован и зафиксирован на стек RapidOCR (`rapidocr==3.9.2`) + ONNX Runtime (`onnxruntime==1.30.0`) + PP-OCRv6 small (детекция и распознавание) и мобильный классификатор ориентации (TASK-008, ADR-015). Реализована динамическая интроспекция активной сессии ONNX Runtime с криптографической проверкой SHA256 хешей файлов весов (`verify_model_stack()`). Эксплицитно задокументирована зависимость от приватного атрибута `_model_path` обертки RapidOCR с гарантией loud failure при его отсутствии. Автоматически экспортируется манифест `experiments/runs/ocr_stack_manifest.json` с четким разделением на секции `configured`, `resolved`, `environment` и структурированный блок `resize_policy` (`configured`, `resolved`, `verified: true`). Зафиксировано строгое методологическое правило ресайза: базисы B0, B1 и B2 используют одну и ту же политику и конфигурацию OCR-ресайза, а физические результирующие размеры могут легитимно различаться из-за различий входных изображений. Полная изоляция от Ground Truth в основном контракте `recognize(image, document_id) -> OCRResult`, отдельный диагностический метод `diagnostic_gt_region_recognition(image, document_id, gt_boxes)`. Детерминированная эвристика порядка чтения (линейная кластеризация по $Y$ с настраиваемым допуском `line_tolerance_factor=0.5` и сортировка слева направо по $X$), сохранение исходных полигонов в `metadata["raw_quadrilaterals"]` и derived `BoundingBox` без клиппинга. Статус оценки на реальном датасете: `REAL-DATA OCR BASELINE: PENDING` (запуск на фикстурах для валидации пайплайна).
- **Evaluation & Metrics:** Реализованы метрики OCR и KIE (`src/evaluation/`):
  - *OCR:* CER, WER и строго определенный **Character-NED similarity** ($1.0 - \frac{\text{edit\_distance}}{\max(\text{len}(pred), \text{len}(gt), 1)}$) в диапазоне $[0.0, 1.0]$. Симметричная нормализация текста OCR (Unicode NFC, схлопывание пробелов, удаление управляющих символов), сохранение 4 представлений текста (`raw_pred`, `raw_gt`, `norm_pred`, `norm_gt`).
  - *KIE:* Реализована двухуровневая система оценки (`src/evaluation/kie_metrics.py`, `KIEEvaluator`):
    - `analytical_metrics`: пополевые Precision, Recall, F1 (как `raw`, так и `normalized`), Macro Precision, Recall, F1 по 4 полям, независимые показатели `raw_doc_em` и `normalized_doc_em`.
    - Нормализация текста `normalize_kie_text` (NFC, lowercase, схлопывание пробелов, удаление граничной пунктуации с сохранением внутренней).
    - Нормализация сумм `normalize_total_amount`: строго evaluator-side протокол (очистка от префиксов `$`, `RM`, `MYR`, нормализация пробелов, консервативная канонизация десятичного представления `10` $\to$ `10.00`, `10.5` $\to$ `10.50`). В raw evaluation действует строгое равенство `prediction == GT`. Данное преобразование категорически запрещено применять на этапе инференса OCR/KIE; политика нормализации заморожена (policy freeze) перед экспериментами.
- **Preprocessing Pipeline & B0/B1/B2 Baselines:** Реализован (`src/preprocessing/`, примитивы P1–P5, `PreprocessingPipeline`, фабрика `get_preprocessor()`, раннер `run_b0_b1_b2_comparison()`, конфигурация `configs/preprocessing.yaml`, CLI `scripts/run_preprocessing_baseline.py`). Полная изоляция от Ground Truth. Текущие параметры зафиксированы как предварительные (**provisional**); статус оценки на реальных данных зафиксирован как `REAL-DATA PREPROCESSING BASELINE: PENDING`.
- **KIE Extractor & Downstream Baseline:** Реализован (`src/kie/`, `RuleBasedKIEEngine`, `MockKIEEngine`, фабрика `get_kie_engine()`, примитивы `src/kie/rules/` для полей `company`, `date`, `address`, `total`, конфигурация `configs/kie.yaml`, CLI раннер `scripts/run_kie_baseline.py`). Строгий контракт без утечки эталона `extract(ocr_result, document_id) -> KIEResult` (аргумент `image` полностью исключен из API). Сохранение аудита происхождения токенов (`metadata["field_provenance"]`). Агрессивный фоллбэк «наибольшее число = total» по умолчанию строго отключен (`fallback_largest_amount: false`). В сквозном раннере принудительно зафиксирован порядок исполнения `ocr_inference` $\to$ `kie_inference` $\to$ `gt_loading` $\to$ `evaluation`. Статус оценки на реальных данных: `PENDING_REAL_DATA`.
- **Experiment Runner:** Не реализован.
- **Kaggle Automation Scripts:** Не реализованы.

---

## 5. Тестирование и покрытие
- Команда запуска тестов:
  ```bash
  pytest -v --cov=src --cov-report=term-missing
  ```
- Результат: **378 тестов пройдены успешно**, суммарное покрытие: **94%** (2036 statements, 128 missed):
  - `src/core`: **100%** (368 statements, 0 missed)
  - `src/datasets`: **100%** (159 statements, 0 missed)
  - `src/degradation`: **98%** (241 statements, 5 missed)
  - `src/ocr`: **88%** (266 statements, 34 missed)
  - `src/kie`: **92%** (294 statements, 23 missed)
  - `src/evaluation`: **97%** (251 statements, 8 missed)
  - `src/preprocessing`: **86%** (338 statements, 46 missed)



