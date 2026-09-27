# TASKS

## Active

*Нет активных задач.*

## Completed

- **TASK-015 — Real-Data Preprocessing Robustness B2 Matrix ($N=126$) on GPU**
  - *Дата завершения:* 2026-09-28
  - *Статус:* Завершена успешно (PASS).
  - *Результат:* Проведен полномасштабный исследовательский эксперимент $B_2$ по исследованию устойчивости и потенциала восстановления (recovery) OCR и downstream KIE при применении фиксированных пайплайнов предобработки после деградаций на валидационном сплите SROIE ($N=126$, seed 42) в целевом аппаратном окружении `CUDAExecutionProvider` (NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM) в строгом соответствии с ADR-011, ADR-012, ADR-017 и ADR-018:
    1. *Матрица и целостность выполнения:*
       - 129 экспериментальных условий: 1 контрольное условие ($D_0\_S_0\_P_0$) + 8 деградаций ($D_1$–$D_8$) $\times$ 4 уровня severity ($S_1$–$S_4$) $\times$ 4 зафиксированных пайплайна предобработки ($P_{\text{minimal}}$, $P_{\text{contrast}}$, $P_{\text{standard}}$, $P_{\text{binarization}}$) = 128 условий $B_2$.
       - Условия $P_0$ для $B_1$ не перегенерировались (`include_b1: false`), а взяты из внешнего замороженного эталона `b1_degraded_validation_n126_gpu` (commit `0c6c1a2b`).
       - Общий объем: $129 \times 126 = 16\,254$ сквозных оценок конвейера `Image -> Degradation -> Preprocessing -> OCR (RapidOCR PP-OCRv6, CUDA) -> Rule-Based KIE -> Evaluation`.
       - Учет сбоев: $16\,254$ успешно завершенных оценок, $0$ отказов (`failure_rate: 0.00%`). Время прогона: $23970.15$ с (~6.66 ч).
       - Изоляция тестового сплита: сплит `test` ($N=347$) строго закрыт guard-флагом и не использовался.
    2. *Верификация контрольного условия ($D_0\_S_0\_P_0$ vs $B_0\_GPU$):*
       - Контрольное условие $D_0\_S_0\_P_0$ внутри $B_2$ численно идентично замороженному baseline $B_0\_GPU$ (`b0_clean_validation_n126_gpu`) с нулевым отклонением по всем метрикам ($\Delta\text{CER} = 0.0000$, $\Delta\text{WER} = 0.0000$, $\Delta\text{NED} = 0.0000$, $\Delta F_1 = 0.0000$, $\Delta\text{Hmean} = 0.0000$, $\Delta\text{DocEM} = 0.0000$).
    3. *Агрегированные результаты по пайплайнам предобработки:*
       - `p_standard_receipt_enhancement` (deskew $\to$ clahe $\to$ denoise):
         - Mean $\Delta\text{CER}_{\text{recovery}} = -0.0103$, улучшил CER в 15 из 32 условий.
         - Показал выраженное восстановление на деградации поворота $D_6$: на $S_1$ $\Delta\text{CER} = -0.0119$, на $S_2$ $\Delta\text{CER} = -0.0969$ ($\Delta F_1 = +0.0278$), на $S_3$ $\Delta\text{CER} = -0.1539$ ($\Delta F_1 = +0.0496$, возврат CER с 0.4812 до 0.3273).
         - Граничный эффект: на $S_4$ (угол 25°) превышен максимальный лимит `max_angle=15.0°`, поэтому поворот не скомпенсирован ($\Delta\text{CER} = -0.0017$).
         - На деградации перспективы $D_7$: средний $\Delta\text{CER}_{\text{recovery}} = -0.0225$.
       - `p_contrast_enhancement` (grayscale $\to$ clahe):
         - Mean $\Delta\text{CER}_{\text{recovery}} = -0.0021$, улучшил CER в 15 из 32 условий.
         - Эффективен против теней $D_8$: на $S_4$ $\Delta\text{CER}_{\text{recovery}} = -0.0272$ (снижение CER с 0.3497 до 0.3225), $\Delta F_1 = +0.0040$.
       - `p_minimal_cleanup` (grayscale $\to$ median denoise ksize=3):
         - Mean $\Delta\text{CER}_{\text{recovery}} = +0.0038$, mean $\Delta F_{1,\text{recovery}} = -0.0061$.
         - В основном имеет нейтральный или слабоотрицательный эффект из-за сглаживания тонких штрихов шрифта на чеках.
       - `p_aggressive_binarization` (grayscale $\to$ adaptive gaussian thresholding):
         - Mean $\Delta\text{CER}_{\text{recovery}} = +0.0886$, ухудшил CER в 31 из 32 условий; ухудшил Macro $F_1$ во всех 32 условиях (mean $\Delta F_1 = -0.0929$).
         - Эмпирически подтверждена деструктивность жесткой бинаризации для нейросетевых OCR-распознавателей (разрушение антиалиасинга и целостности штрихов).
    4. *Артефакты эксперимента:*
       - `experiments/runs/b2_preprocessing_validation_n126_gpu/summary.json`
       - `experiments/runs/b2_preprocessing_validation_n126_gpu/manifest.json`
       - `experiments/runs/b2_preprocessing_validation_n126_gpu/per_document.jsonl` (16 254 записи)
       - `experiments/runs/b2_preprocessing_validation_n126_gpu/b2_conditions_detailed.csv` (129 строк)

- **TASK-014 — Real-Data Degradation Baseline B1 on SROIE Validation Split ($N=126$) on GPU**
  - *Дата завершения:* 2026-09-27
  - *Статус:* Завершена успешно (PASS).
  - *Результат:* Проведен полномасштабный фундаментальный исследовательский эксперимент $B_1$ по измерению деградационной устойчивости OCR и downstream KIE на каноническом валидационном сплите SROIE ($N=126$, seed 42) в целевом аппаратном окружении `CUDAExecutionProvider` (NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM) в строгом соответствии с ADR-011, ADR-012 и ADR-017:
    1. *Матрица и целостность выполнения:*
       - 33 экспериментальных условия: 8 деградаций ($D_1$–$D_8$) $\times$ 4 уровня severity ($S_1$–$S_4$) $\times$ $P_0$ (32 условия) + 1 контрольное условие ($D_0\_S_0\_P_0$).
       - Общий объем: $33 \times 126 = 4\,158$ сквозных оценок конвейера `Image -> Degradation -> OCR (RapidOCR PP-OCRv6, CUDA) -> Rule-Based KIE -> Evaluation`.
       - Учет сбоев: $4\,158$ успешно завершенных прогонов, $0$ отказов (`failure_rate: 0.00%`). Общее время: $6842.44$ с (~1.9 ч).
       - Изоляция тестового сплита: сплит `test` ($N=347$) строго закрыт guard-флагом и не затрагивался.
    2. *Верификация контрольного условия ($D_0\_S_0\_P_0$ vs $B_0\_GPU$):*
       - Контрольное условие $D_0\_S_0\_P_0$ точно воспроизвело замороженный baseline $B_0\_GPU$ (`b0_clean_validation_n126_gpu`) с нулевым отклонением по всем метрикам:
         - $\Delta\text{CER} = +0.000000$ ($\text{CER} = 0.3203$)
         - $\Delta\text{WER} = +0.000000$ ($\text{WER} = 0.4831$)
         - $\Delta\text{Char-NED} = +0.000000$ ($\text{NED} = 0.6816$)
         - $\Delta\text{Macro } F_1 = +0.000000$ ($\text{Macro } F_1 = 0.4782$)
         - $\Delta\text{Entity Hmean} = +0.000000$ ($\text{Hmean} = 0.4979$)
    3. *Итоговая матрица метрик B1 (33 условия):*
       | Условие | Деградация | Sev | CER (norm) | $\Delta$CER | WER (norm) | Char-NED | Macro $F_1$ | $\Delta F_1$ | Entity Hmean | $\Delta$Hmean |
       |---|---|---|---|---|---|---|---|---|---|---|
       | **D0_S0_P0** | **Control (B0)** | **S0** | **0.3203** | **+0.0000** | **0.4831** | **0.6816** | **0.4782** | **+0.0000** | **0.4979** | **+0.0000** |
       | D1_S1_P0 | Gaussian Blur | S1 | 0.3205 | +0.0002 | 0.4822 | 0.6819 | 0.4861 | +0.0079 | 0.5052 | +0.0073 |
       | D1_S2_P0 | Gaussian Blur | S2 | 0.3286 | +0.0083 | 0.5062 | 0.6731 | 0.4702 | -0.0080 | 0.4876 | -0.0103 |
       | D1_S3_P0 | Gaussian Blur | S3 | 0.5203 | +0.2000 | 0.6938 | 0.4806 | 0.3472 | -0.1310 | 0.3968 | -0.1011 |
       | D1_S4_P0 | Gaussian Blur | S4 | 0.8608 | +0.5405 | 0.9029 | 0.1395 | 0.1171 | -0.3611 | 0.1850 | -0.3129 |
       | D2_S1_P0 | Motion Blur | S1 | 0.3335 | +0.0132 | 0.5276 | 0.6678 | 0.4583 | -0.0199 | 0.4772 | -0.0207 |
       | D2_S2_P0 | Motion Blur | S2 | 0.3858 | +0.0655 | 0.6169 | 0.6150 | 0.4206 | -0.0576 | 0.4412 | -0.0567 |
       | D2_S3_P0 | Motion Blur | S3 | 0.6321 | +0.3118 | 0.8602 | 0.3681 | 0.1726 | -0.3056 | 0.2167 | -0.2812 |
       | D2_S4_P0 | Motion Blur | S4 | 0.8027 | +0.4824 | 0.9320 | 0.1973 | 0.0853 | -0.3929 | 0.1243 | -0.3736 |
       | D3_S1_P0 | Gaussian Noise | S1 | 0.3204 | +0.0001 | 0.4713 | 0.6820 | 0.4980 | +0.0198 | 0.5138 | +0.0159 |
       | D3_S2_P0 | Gaussian Noise | S2 | 0.3259 | +0.0056 | 0.4872 | 0.6769 | 0.4802 | +0.0020 | 0.4949 | -0.0030 |
       | D3_S3_P0 | Gaussian Noise | S3 | 0.3569 | +0.0366 | 0.5409 | 0.6453 | 0.4504 | -0.0278 | 0.4734 | -0.0245 |
       | D3_S4_P0 | Gaussian Noise | S4 | 0.4134 | +0.0931 | 0.6150 | 0.5888 | 0.4286 | -0.0496 | 0.4576 | -0.0403 |
       | D4_S1_P0 | JPEG Compression | S1 | 0.3215 | +0.0012 | 0.4879 | 0.6803 | 0.4821 | +0.0039 | 0.5020 | +0.0041 |
       | D4_S2_P0 | JPEG Compression | S2 | 0.3233 | +0.0030 | 0.4966 | 0.6783 | 0.4663 | -0.0119 | 0.4851 | -0.0128 |
       | D4_S3_P0 | JPEG Compression | S3 | 0.3237 | +0.0034 | 0.4989 | 0.6781 | 0.4722 | -0.0060 | 0.4928 | -0.0051 |
       | D4_S4_P0 | JPEG Compression | S4 | 0.3378 | +0.0175 | 0.5368 | 0.6640 | 0.4583 | -0.0199 | 0.4797 | -0.0182 |
       | D5_S1_P0 | Downsampling | S1 | 0.3214 | +0.0011 | 0.4844 | 0.6811 | 0.4802 | +0.0020 | 0.4990 | +0.0011 |
       | D5_S2_P0 | Downsampling | S2 | 0.3213 | +0.0010 | 0.4862 | 0.6807 | 0.4702 | -0.0080 | 0.4907 | -0.0072 |
       | D5_S3_P0 | Downsampling | S3 | 0.3358 | +0.0155 | 0.5239 | 0.6659 | 0.4583 | -0.0199 | 0.4787 | -0.0192 |
       | D5_S4_P0 | Downsampling | S4 | 0.4498 | +0.1295 | 0.6978 | 0.5512 | 0.3056 | -0.1726 | 0.3434 | -0.1545 |
       | D6_S1_P0 | Rotation | S1 | 0.3314 | +0.0111 | 0.4903 | 0.6709 | 0.4762 | -0.0020 | 0.4948 | -0.0031 |
       | D6_S2_P0 | Rotation | S2 | 0.4180 | +0.0977 | 0.5771 | 0.5846 | 0.4385 | -0.0397 | 0.4604 | -0.0375 |
       | D6_S3_P0 | Rotation | S3 | 0.4812 | +0.1609 | 0.6450 | 0.5210 | 0.4048 | -0.0734 | 0.4224 | -0.0755 |
       | D6_S4_P0 | Rotation | S4 | 0.5577 | +0.2374 | 0.7148 | 0.4436 | 0.3452 | -0.1330 | 0.3651 | -0.1328 |
       | D7_S1_P0 | Perspective | S1 | 0.3337 | +0.0134 | 0.4940 | 0.6687 | 0.4802 | +0.0020 | 0.5011 | +0.0032 |
       | D7_S2_P0 | Perspective | S2 | 0.3761 | +0.0558 | 0.5343 | 0.6262 | 0.4464 | -0.0318 | 0.4663 | -0.0316 |
       | D7_S3_P0 | Perspective | S3 | 0.4320 | +0.1117 | 0.5878 | 0.5701 | 0.4385 | -0.0397 | 0.4585 | -0.0394 |
       | D7_S4_P0 | Perspective | S4 | 0.4960 | +0.1757 | 0.6566 | 0.5058 | 0.4067 | -0.0715 | 0.4271 | -0.0708 |
       | D8_S1_P0 | Shadow | S1 | 0.3200 | -0.0003 | 0.4828 | 0.6820 | 0.4841 | +0.0059 | 0.5026 | +0.0047 |
       | D8_S2_P0 | Shadow | S2 | 0.3226 | +0.0023 | 0.4916 | 0.6793 | 0.4841 | +0.0059 | 0.5036 | +0.0057 |
       | D8_S3_P0 | Shadow | S3 | 0.3217 | +0.0014 | 0.4880 | 0.6799 | 0.4841 | +0.0059 | 0.5010 | +0.0031 |
       | D8_S4_P0 | Shadow | S4 | 0.3497 | +0.0294 | 0.5213 | 0.6518 | 0.4762 | -0.0020 | 0.4948 | -0.0031 |
    4. *Упорядочение экспериментальных условий при Severity S4 по изменению метрик относительно $B_0$:*
       - **По возрастанию величины $\Delta\text{CER}$ (увеличение посимвольной ошибки распознавания):**
         1. $D_1$ Gaussian Blur: $\Delta\text{CER} = +0.5405$ ($\text{CER} = 0.8608$, $\text{NED} = 0.1395$)
         2. $D_2$ Motion Blur: $\Delta\text{CER} = +0.4824$ ($\text{CER} = 0.8027$, $\text{NED} = 0.1973$)
         3. $D_6$ Rotation: $\Delta\text{CER} = +0.2374$ ($\text{CER} = 0.5577$, $\text{NED} = 0.4436$)
         4. $D_7$ Perspective: $\Delta\text{CER} = +0.1757$ ($\text{CER} = 0.4960$, $\text{NED} = 0.5058$)
         5. $D_5$ Downsampling: $\Delta\text{CER} = +0.1295$ ($\text{CER} = 0.4498$, $\text{NED} = 0.5512$)
         6. $D_3$ Gaussian Noise: $\Delta\text{CER} = +0.0931$ ($\text{CER} = 0.4134$, $\text{NED} = 0.5888$)
         7. $D_8$ Shadow: $\Delta\text{CER} = +0.0294$ ($\text{CER} = 0.3497$, $\text{NED} = 0.6518$)
         8. $D_4$ JPEG Compression: $\Delta\text{CER} = +0.0175$ ($\text{CER} = 0.3378$, $\text{NED} = 0.6640$)
       - **По величине $\Delta\text{Macro } F_1$ (убывание пополевого $F_1$):**
         1. $D_2$ Motion Blur: $\Delta F_1 = -0.3929$ ($F_1 = 0.0853$, $\text{Hmean} = 0.1243$)
         2. $D_1$ Gaussian Blur: $\Delta F_1 = -0.3611$ ($F_1 = 0.1171$, $\text{Hmean} = 0.1850$)
         3. $D_5$ Downsampling: $\Delta F_1 = -0.1726$ ($F_1 = 0.3056$, $\text{Hmean} = 0.3434$)
         4. $D_6$ Rotation: $\Delta F_1 = -0.1330$ ($F_1 = 0.3452$, $\text{Hmean} = 0.3651$)
         5. $D_7$ Perspective: $\Delta F_1 = -0.0715$ ($F_1 = 0.4067$, $\text{Hmean} = 0.4271$)
         6. $D_3$ Gaussian Noise: $\Delta F_1 = -0.0496$ ($F_1 = 0.4286$, $\text{Hmean} = 0.4576$)
         7. $D_4$ JPEG Compression: $\Delta F_1 = -0.0199$ ($F_1 = 0.4583$, $\text{Hmean} = 0.4797$)
         8. $D_8$ Shadow: $\Delta F_1 = -0.0020$ ($F_1 = 0.4762$, $\text{Hmean} = 0.4948$)
    5. *Эмпирические показатели извлечения сущностей KIE при S4:*
       - **Date:** сохраняет наивысшие значения $F_1$ среди 4 сущностей при $D_3, D_4, D_6, D_7, D_8$ ($F_1 \in [0.86, 0.97]$); при $D_1$ и $D_2$ зафиксировано снижение до $F_1 = 0.3311$ и $0.2083$ соответственно.
       - **Total:** принимает значения $F_1 \in [0.61, 0.62]$ при $D_3, D_4, D_8$ и более низкие значения при $D_6$ ($0.3192$), $D_7$ ($0.4240$), $D_1$ ($0.2119$) и $D_2$ ($0.2266$).
       - **Company:** принимает значения $F_1 \in [0.38, 0.41]$ при $D_3, D_4, D_7, D_8$, $0.3267$ при $D_5$, $0.2063$ при $D_6$, $0.1758$ при $D_1$ и $0.1005$ при $D_2$.
       - **Address:** во всех условиях сохраняет значения $F_1 \le 0.026$ (исходный уровень в $B_0$: $F_1 = 0.0079$).
    6. *Артефакты:* Сохранены в `experiments/runs/b1_degraded_validation_n126_gpu/` (`summary.json`, `manifest.json`, `per_document.jsonl` — 4,158 строк, `ocr_stack_manifest.json`, `config.yaml`, `b1_conditions_detailed.csv`). Все 406 тестов проекта пройдены успешно (100% pass).


- **TASK-013-GPU — Real-Data Clean Baseline B0 on SROIE Validation Split ($N=126$) on GPU**
  - *Дата завершения:* 2026-09-27
  - *Статус:* Завершена успешно (PASS).
  - *Результат:* Сформирована фундаментальная контрольная точка исследования в целевом аппаратном окружении `CUDAExecutionProvider` (NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM) — чистый бейзлайн $B_0$ (`D0_S0_P0`) на валидационном сплите SROIE ($N=126$, seed 42) в соответствии с ADR-017:
    1. *Канонический сквозной конвейер на GPU:* Выполнен пайплайн `SROIE Image -> D0 (identity) -> P0 (identity) -> RapidOCR (PP-OCRv6 small, CUDAExecutionProvider) -> Rule-Based KIE -> Evaluation`.
    2. *Учет сбоев:* 126 из 126 документов обработаны успешно (`success: 126, failed: 0, failure_rate: 0.00%`). Время выполнения: 172.48 с (~2.8 мин).
    3. *Метрики OCR (N=126, bootstrap 95% CI):*
       - CER Normalized: $0.3203$ (CI: $[0.2993, 0.3421]$, median: $0.3397$) — идентично CPU ($0.3203$)
       - WER Normalized: $0.4831$ (CI: $[0.4570, 0.5071]$, median: $0.4903$) — $\Delta = -0.0002$ относительно CPU
       - Character-NED Normalized: $0.6816$ (CI: $[0.6592, 0.7019]$, median: $0.6629$) — идентично CPU ($0.6816$)
    4. *Метрики KIE (N=126):*
       - Macro F1 Normalized: $0.4782$ (CI: $[0.4444, 0.5079]$)
       - Entity Hmean (Task-3 Micro): $0.4979$ (241 извлеченная сущность из 504 — идентично CPU)
       - Совпадение KIE-полей с CPU B0: 100% (`company`: 126/126, `date`: 126/126, `address`: 126/126, `total`: 126/126)
    5. *Сравнение с B0 CPU:* На всех 126 документах подтверждена 100% эквивалентность нормализованного текста (126/126 exact matches, mean CER = 0.0000, mean WER = 0.0000, mean NED = 1.0000). Артефакт сравнения: `experiments/benchmarks/b0_cpu_vs_gpu_comparison.json`.
    6. *Артефакты B0 GPU:* Сформированы в `experiments/runs/b0_clean_validation_n126_gpu/` (`config.yaml`, `manifest.json`, `ocr_stack_manifest.json`, `per_document.jsonl`, `summary.json`, `logs/run.log`). Хеш конфигурации: `63bf882aa172...`, статус данных: `REAL_SROIE`.

- **TASK-013 — Real-Data Clean Baseline B0 on SROIE Validation Split ($N=126$) (CPU)**
  - *Дата завершения:* 2026-09-27
  - *Статус:* Завершена успешно (PASS). Сохранен как исторический референс CPU.


- **TASK-012 — Real-Data Degradation Calibration on Validation Split ($N=126$)**
  - *Дата завершения:* 2026-09-27
  - *Статус:* Завершена успешно (PASS).
  - *Результат:* Проведена окончательная эмпирическая калибровка физических параметров и метрик искажений для всех 8 семейств синтетических деградаций ($D_1$–$D_8$) по 5 уровням интенсивности (severities $0$–$4$) на канонической валидационной выборке SROIE ($N=126$, seed 42) в строгом соответствии с ADR-011 и ADR-012:
    1. *Защита данных и изоляция:* Калибровка выполнена исключительно на 126 валидационных изображениях (`data/SROIE2019/train/img`, индексы 500:626, seed 42). Тестовый сплит ($N=347$) полностью изолирован и не затрагивался. Никакая эталонная разметка (OCR/KIE GT) в процедуре калибровки не использовалась.
    2. *Оптимизация и устранение OOM:*
       - Обнаружены аномально большие сканы высокого разрешения (до 35 Мп, $7016 \times 4961$), вызывавшие скачки памяти до 7.2 ГБ при наивном предварительном кэшировании.
       - Внедрено ограничение максимального размера `max_side=2000` в `load_single_image` в строгом соответствии с политикой разрешения RapidOCR (ADR-015).
       - Переработан цикл калибровки на потоковую обработку документов (одно изображение в памяти за раз), что снизило потребление RAM до стабильных 80–140 МБ.
       - Расчеты MSE и MAE оптимизированы через `cv2.norm(..., NORM_L2SQR)` и `cv2.norm(..., NORM_L1)` (ускорение в 42 раза).
       - Расчет SSIM ограничен сценариями, где он является целевой метрикой ($D_4$ JPEG и $D_5$ Downsampling), исключая вычислительно бессмысленные расчеты SSIM для поворотов, теней и перспективных искажений.
    3. *Монотонность параметров и метрик:* Для всех 8 семейств деградаций подтверждена 100% монотонность как параметров, так и объективных метрик искажений на реальных данных ($1 \to 4$):
       - $D_1$ Gaussian Blur: PSNR 27.75 $\to$ 22.81 $\to$ 20.68 $\to$ 19.88 dB (Laplacian ratio 0.14 $\to$ 0.00).
       - $D_2$ Motion Blur: PSNR 24.67 $\to$ 22.29 $\to$ 20.52 $\to$ 19.85 dB (Laplacian ratio 0.17 $\to$ 0.04).
       - $D_3$ Gaussian Noise: PSNR 29.94 $\to$ 22.45 $\to$ 17.60 $\to$ 13.94 dB (MAE 5.59 $\to$ 32.24).
       - $D_4$ JPEG Compression: PSNR 43.11 $\to$ 36.46 $\to$ 32.92 $\to$ 29.28 dB (SSIM 0.99 $\to$ 0.92).
       - $D_5$ Downsampling: PSNR 29.67 $\to$ 27.20 $\to$ 23.86 $\to$ 21.35 dB (SSIM 0.97 $\to$ 0.82).
       - $D_6$ Rotation: MAE 15.88 $\to$ 18.52 $\to$ 19.98 $\to$ 20.79.
       - $D_7$ Perspective Distortion: scale 0.05 $\to$ 0.10 $\to$ 0.18 $\to$ 0.28.
       - $D_8$ Shadow: Luminance drop 0.09 $\to$ 0.23 $\to$ 0.43 $\to$ 0.68 (MAE 21.19 $\to$ 161.16).
    4. *Артефакты:* Сгенерированы валидные артефакты `experiments/calibration/calibration_config.yaml` и `experiments/calibration/calibration_report.json` со статусом `parameter_status: final_research_calibrated`, `real_data_calibration: COMPLETED`, `calibration_dataset: sroie_validation`, `num_calibration_images: 126`. Обновлен `configs/degradation.yaml`.
    5. *Тестирование:* Полный тестовый набор (`pytest -v --cov=src`) проходит со 100% успехом (396 passed, 0 failed, 92% coverage).

- **TASK-011-A-R2 — Test Suite & Baseline Runners Isolation Resolution**
  - *Дата завершения:* 2026-09-27
  - *Статус:* Завершена успешно (PASS).
  - *Результат:* Устранены зависания и ошибки тестов после загрузки реального датасета SROIE:
    1. *Калибровочный раннер:* Исправлена функция `find_calibration_images` в `scripts/calibrate_degradations.py`, исключено неявное переключение на 126 изображений реального валидационного сплита при отсутствии флага `--data-root` / `--allow-real-data`. Оптимизированы расчеты метрик искажений с `float64` до `float32`. Добавлен регрессионный тест `test_calibration_fixture_contract_and_performance`.
    2. *KIE и OCR Baseline раннеры:* Исправлен механизм определения корня датасета в `scripts/run_kie_baseline.py`, `scripts/run_ocr_baseline.py` и `scripts/run_preprocessing_baseline.py`. При передаче аргумента `data_root` (в частности фикстур `tests/fixtures/sroie_valid`) раннеры больше не подменяют его silently полным корпусом `data/SROIE2019`.
    3. *SROIE Ground Truth парсер координат:* Скорректирована эвристика проверки 8 координат в `src/datasets/sroie.py` и `scripts/audit_sroie.py`: теперь проверяется наличие двух числовых координат (токенов 9 и 10 при длине строки $\ge 11$), что предотвращает ложное отбрасывание квитанций с номерами домов/улиц (например, `13, JLN TASIK UTAMA 8`).
    4. *Устойчивость и мониторинг `UnifiedExperimentRunner`:* Добавлен флаг `--debug-timing`, небуферизованный вывод логов и инкрементальный сброс (`flush()`) каждой записи в `per_document.jsonl`.
    5. *Тестирование:* Полный тестовый набор (`pytest -v --cov=src`) проходит со 100% успехом (396 passed, 0 failed, coverage 92%).

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
