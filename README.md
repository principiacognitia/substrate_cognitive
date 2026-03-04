# Principia Cognitia: Substrate-Independent Cognitive Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-stage--2--development-orange)](./stage2/README.md)

## 📖 Описание

**Principia Cognitia** — это исследовательский фреймворк для моделирования когнитивных систем, способных функционировать на различных субстратах (биологических и искусственных). Архитектура основана на принципах активного вывода (Active Inference), реологической динамики контроля и мета-когнитивного арбитража между модель-свободными (MF) и модель-основанными (MB) режимами.

### Ключевые компоненты

| Компонент | Описание | Статус |
| :--- | :--- | :--- |
| **S-O-R Primitive** | Базовые единицы: States, Operations, Rules | ✅ Завершено |
| **Multi-Gate** | Арбитраж между MF/MB режимами | ✅ Завершено |
| **Rheology (V_G, V_p)** | Вязкость контроля и действий | ✅ Завершено |
| **Two-Step Task** | Валидация MB/MF сигнатур | 🟡 В разработке |
| **Reversal Task** | Валидация когнитивной ригидности | 🟡 В разработке |
| **Absence Check** | Инференс отсутствия | ⬜ Запланировано |
| **Trauma Learning** | One-shot обучение | ⬜ Запланировано |

## 🗂️ Структура репозитория

```
substrate_cognitive/
├── mvp/                    # Завершённый MVP (T-maze, v5.0)
├── stage2/                 # Текущая разработка (Two-Step + Reversal)
│   ├── core/               # Общие модули (gate, rheology, baselines)
│   ├── twostep/            # Two-Step Task (Daw et al., 2011)
│   ├── reversal/           # Block-Reversal Task
│   └── analysis/           # Скрипты анализа и визуализации
├── stage3_survival/        # Запланировано (Trauma, Absence Check)
├── docs/                   # Документация (mapping.md, specs)
├── logs/                   # Экспериментальные логи (игнорируются git)
└── tests/                  # Автоматические тесты
```

## 🚀 Быстрый старт

```bash
# Клонирование
git clone https://github.com/YOUR_USERNAME/substrate_cognitive.git
cd substrate_cognitive

# Установка зависимостей
pip install -r requirements.txt

# Запуск Stage 2 экспериментов
python -m stage2.twostep.debug_sanity_check
python -m stage2.twostep.debug_ablation
python -m stage2.reversal.run_reversal

# Генерация графиков для статьи
python -m stage2.analysis.generate_all_figures
```

## 📊 Текущий статус (Stage 2)

| Метрика | Результат | Статус |
| :--- | :--- | :--- |
| **MB/MF сигнатуры** | interaction coef = 0.47, p < 0.001 | ✅ PASS |
| **V_G гистерезис** | Latency = 35 триалов (медиана) | ✅ PASS |
| **Абляция V_G** | 37× разница в латентности, p = 4.47e-10 | ✅ PASS |
| **Reversal Task** | В разработке | 🟡 IN PROGRESS |
| **CV Log-Likelihood** | Запланировано | ⬜ TODO |

## 📄 Документация

- [Stage 2 Overview](./stage2/README.md) — Детали текущей разработки
- [Two-Step Task](./stage2/twostep/README.md) — Спецификация среды и агента
- [Reversal Task](./stage2/reversal/README.md) — Спецификация реверсального обучения
- [Mapping Document](./docs/mapping.md) — Соответствие латентов и наблюдаемых метрик

## 🔬 Публикации

1. **В подготовке (2026):** *Gate-Rheology: Inertia of Cognitive Control Explains Meta-Rigidity in Sequential Decision Making and Reversal Learning.* (Препринт на bioRxiv)

## 🤝 Вклад

Этот репозиторий является частью диссертационного проекта **Principia Cognitia**. По вопросам сотрудничества обращайтесь к автору.

## 📝 Лицензия

MIT License — см. [LICENSE](LICENSE) файл.

## 📚 Ключевые источники

- Daw, N. D., et al. (2011). Model-based influences on humans' choices and striatal prediction errors. *Neuron, 69*(6), 1204–1215.
- Le, N. M., et al. (2023). Mixtures of strategies underlie rodent behavior during reversal learning. *PLOS Computational Biology, 19*(9), e1011430.
- Hasz, B. M., & Redish, A. D. (2018). Deliberation and procedural automation on a two-step task for rats. *Frontiers in Integrative Neuroscience, 12*, 30.
