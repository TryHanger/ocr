# Исследование качества скана и устойчивости OCR/KIE к деградациям

Исследовательский проект, направленный на количественную оценку влияния различных типов и степеней деградации качества изображений документов на точность распознавания текста (OCR) и последующего извлечения ключевых данных (Key Information Extraction, KIE), а также на оценку эффективности методов предварительной обработки (preprocessing).

---

## Экспериментальный конвейер (Pipeline)

```text
Исходный документ
        ↓
Генерация деградации (Blur, Noise, Skew, Shadow, Compression, Resolution)
        ↓
Preprocessing / Enhancement (Deskew, Denoise, CLAHE, Binarization)
        ↓
OCR Engine (Извлечение токенов и координат)
        ↓
KIE Engine (Семантическое извлечение ключевых полей)
        ↓
Evaluation (CER, WER, Field F1, Document Exact Match)
        ↓
Robustness Analysis (Построение кривых устойчивости и хитмапов)
```

---

## Структура проекта

```text
ocr/
├── configs/            # YAML-конфигурации экспериментов
├── data/               # Датасеты и разметка (не коммитятся в git)
├── experiments/        # Результаты экспериментов (runs/ и figures/)
├── kaggle/             # Скрипты и конфигурации для запуска на Kaggle Kernels
├── scripts/            # CLI-утилиты для запуска экспериментов
├── src/                # Исходный код системы
│   ├── degradation/    # Модуль деградации изображений
│   ├── preprocessing/  # Модуль улучшения качества
│   ├── ocr/            # Слой абстракции OCR-движков
│   ├── kie/            # Слой извлечения ключевой информации
│   ├── evaluation/     # Модуль метрик и нормализации
│   └── visualization/  # Построение графиков и отчётов
├── tests/              # Автоматизированные тесты
├── CLAUDE.md           # Протокол управления проектом
├── TASKS.md            # Журнал задач (Active / Completed)
├── IMPLEMENTATION.md   # Фактическое состояние реализации
├── DECISIONS.md        # Журнал принятых архитектурных решений (ADR)
├── pyproject.toml      # Конфигурация проекта и зависимостей
├── requirements.txt    # Зависимости базового окружения
└── README.md           # Документация проекта
```

---

## Установка и окружение

### 1. Клонирование репозитория
```bash
git clone https://github.com/TryHanger/ocr.git
cd ocr
```

### 2. Создание виртуального окружения
```bash
python -m venv .venv
# Для Windows:
.venv\Scripts\activate
# Для Linux/macOS:
source .venv/bin/activate
```

### 3. Установка базовых зависимостей
```bash
pip install -r requirements.txt
```

---

## Запуск тестов

Для проверки работоспособности окружения и корректности модулей запустите `pytest`:

```bash
pytest
```
